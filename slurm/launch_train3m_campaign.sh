#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/train3m_environment.sh"
exec bash "$SCRIPT_DIR/launch_engineer_campaign.sh" "$@"
