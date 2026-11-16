#!/usr/bin/env bash
# Compatible with Bash 3.2+ (including the macOS system Bash).
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
port=8000
image=yad2-api:local
name=yad2-api
env_file=.env
encoder_path=
detach=false
dry_run=false

usage() {
    cat <<'EOF'
Usage: bash scripts/docker-build-and-run.sh [options]
  --port PORT           Host port (default: 8000)
  --image IMAGE         Image tag (default: yad2-api:local)
  --name NAME           Container name (default: yad2-api)
  --env-file FILE       Env file, relative to project root (default: .env)
  --encoder-path DIR    Explicit host E5 encoder directory
  --detach              Run in the background
  --dry-run             Print commands without building or starting a container
  --help                Show this help
EOF
}
fail() { printf '%s\n' "$*" >&2; exit 1; }
project_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$project_root" "$1" ;;
    esac
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port|--image|--name|--env-file|--encoder-path)
            [[ $# -ge 2 && -n "$2" ]] || fail "Missing value for $1"
            case "$1" in
                --port) port=$2 ;;
                --image) image=$2 ;;
                --name) name=$2 ;;
                --env-file) env_file=$2 ;;
                --encoder-path) encoder_path=$2 ;;
            esac
            shift 2 ;;
        --detach) detach=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; fail "Unknown option: $1" ;;
    esac
done
[[ "$port" =~ ^[0-9]{1,5}$ ]] || fail 'Port must be an integer between 1 and 65535.'
port=$((10#$port))
(( port >= 1 && port <= 65535 )) || fail 'Port must be between 1 and 65535.'
env_file="$(project_path "$env_file")"
[[ -f "$env_file" ]] || fail "Environment file not found: $env_file. Copy .env.example to .env first."

# Parse only mount-related settings. Never source the file or print its contents.
extractor_enabled=false
configured_encoder=models/multilingual-e5-small
while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    if [[ "$line" =~ ^[[:space:]]*(EXTRACTOR_ENABLED|E5_MODEL_PATH)=(.*)$ ]]; then
        key="${BASH_REMATCH[1]}"
        value="${BASH_REMATCH[2]}"
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"
        case "$key" in
            EXTRACTOR_ENABLED) extractor_enabled="$value" ;;
            E5_MODEL_PATH) configured_encoder="$value" ;;
        esac
    fi
done < "$env_file"
case "$extractor_enabled" in
    [Tt][Rr][Uu][Ee]|1) extractor_enabled=true ;;
    [Ff][Aa][Ll][Ss][Ee]|0) extractor_enabled=false ;;
    *) fail 'EXTRACTOR_ENABLED must be an unquoted true, false, 1, or 0 in the Docker env file.' ;;
esac

run_args=(run --rm --name "$name" -p "$port:8000" --env-file "$env_file")
if "$detach"; then run_args+=(-d); fi
if "$extractor_enabled"; then
    candidates=()
    if [[ -n "$encoder_path" ]]; then
        candidates+=("$(project_path "$encoder_path")")
    else
        candidates+=("$(project_path "$configured_encoder")" "$project_root/models/multilingual-e5-small")
        hub_cache="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
        candidates+=("$hub_cache/models--intfloat--multilingual-e5-small/snapshots/614241f622f53c4eeff9890bdc4f31cfecc418b3")
    fi
    encoder_directory=
    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate/config.json" && -f "$candidate/modules.json" && -f "$candidate/model.safetensors" ]]; then
            encoder_directory="$(cd "$candidate" && pwd -P)"
            break
        fi
    done
    [[ -n "$encoder_directory" ]] || fail 'E5 encoder not found. See docs/optional-parser.md, or pass --encoder-path <directory>.'
    mount_source="$encoder_directory"
    container_encoder=/models/e5
    snapshot_parent="$(dirname "$encoder_directory")"
    repository_directory="$(dirname "$snapshot_parent")"
    # HF snapshot files link to ../../blobs; preserve that layout inside Docker.
    if [[ "$(basename "$snapshot_parent")" == snapshots && -d "$repository_directory/blobs" ]]; then
        mount_source="$repository_directory"
        container_encoder="/models/e5/snapshots/$(basename "$encoder_directory")"
    fi
    [[ "$mount_source" != *,* ]] || fail 'Docker mount paths must not contain commas.'
    printf 'E5 encoder: %s\n' "$encoder_directory"
    run_args+=(--mount "type=bind,source=$mount_source,target=/models/e5,readonly" -e "E5_MODEL_PATH=$container_encoder")
fi
run_args+=("$image")
build_args=(build -t "$image" "$project_root")

if "$dry_run"; then
    printf 'docker'; printf ' %q' "${build_args[@]}"; printf '\n'
    printf 'docker'; printf ' %q' "${run_args[@]}"; printf '\n'
    exit 0
fi
command -v docker >/dev/null 2>&1 || fail 'Docker CLI not found. Install and start Docker first.'
docker info --format '{{.ServerVersion}}' >/dev/null 2>&1 || fail 'Docker daemon is unavailable. Start Docker first.'
docker "${build_args[@]}" || fail 'Docker build failed; the container was not started.'
printf 'API: http://localhost:%s/docs\nContainer: %s. Stop it with: docker stop %s\n' "$port" "$name" "$name"
docker "${run_args[@]}" || fail "Docker run failed. Check the error above; if '$name' already exists, stop it or choose a different --name."
