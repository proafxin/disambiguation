#!/usr/bin/env bash
#
# Upgrade every dependency AND refresh the version constraints in pyproject.toml,
# without touching them by hand.
#
# Strategy
#   1. `uv lock --upgrade` bumps every package in uv.lock to the latest version
#      allowed by the current constraints.
#   2. Diff uv.lock before/after to find which packages actually changed.
#   3. For the changed packages that are *direct* dependencies (declared in
#      pyproject.toml), re-`uv add` them pinned to the freshly-locked version.
#      This rewrites their lower bound in pyproject.toml to the latest compatible
#      version, preserving extras, environment markers and dependency-group /
#      optional-extra membership.
#   4. `uv sync` materialises the environment.
#
# Why `uv add` and not `uv remove` + `uv add`:
#   `uv remove <pkg>` also drops the matching [tool.uv.sources] entry (e.g. the
#   pytorch-cu130 index pin for torch). Re-adding alone would then resolve from
#   the default index and break that setup. Re-`uv add`-ing an existing dependency
#   updates only its version specifier and leaves the source/index intact.

set -euo pipefail

# Resolve project root (parent of this script's directory) so the script works
# regardless of where it is invoked from.
# SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# cd "${PROJECT_ROOT}"

LOCK_FILE="uv.lock"

if [[ ! -f "${LOCK_FILE}" ]]; then
  echo "==> No uv.lock found; creating one first"
  uv lock
fi

# Print "normalized-name<TAB>version" for every package in the given lockfile.
dump_lock_versions() {
  uv run --no-project python - "$1" <<'PY'
import re, sys, tomllib

with open(sys.argv[1], "rb") as fh:
    data = tomllib.load(fh)
for pkg in data.get("package", []):
    name = re.sub(r"[-_.]+", "-", pkg["name"]).lower()  # PEP 503 normalization
    print(f"{name}\t{pkg.get('version', '')}")
PY
}

BEFORE_VERSIONS="$(mktemp)"
AFTER_VERSIONS="$(mktemp)"
trap 'rm -f "${BEFORE_VERSIONS}" "${AFTER_VERSIONS}"' EXIT

echo "==> Recording current locked versions"
dump_lock_versions "${LOCK_FILE}" | sort > "${BEFORE_VERSIONS}"

echo "==> uv lock --upgrade"
uv lock --upgrade

echo "==> Recording new locked versions"
dump_lock_versions "${LOCK_FILE}" | sort > "${AFTER_VERSIONS}"

# Build the plan: for each direct dependency whose locked version changed, emit
# "<scope><TAB><spec>" where scope is one of: main | group:<name> | optional:<extra>
# and spec is "<name>[extras]>=<newversion>[; marker]".
echo "==> Computing changed direct dependencies"
mapfile -t PLAN < <(uv run --no-project python - "${BEFORE_VERSIONS}" "${AFTER_VERSIONS}" <<'PY'
import re, sys, tomllib

def load(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            name, _, ver = line.partition("\t")
            out[name] = ver
    return out

before = load(sys.argv[1])
after = load(sys.argv[2])

def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()

# Packages whose locked version changed (or newly appeared).
changed = {n: v for n, v in after.items() if before.get(n) != v}

with open("pyproject.toml", "rb") as fh:
    pp = tomllib.load(fh)

# Leading "<name>" and optional "[extra1,extra2]" of a PEP 508 requirement.
NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?")

def emit(scope, reqs):
    for req in reqs:
        if not isinstance(req, str):
            continue  # e.g. dependency-group {include-group = "..."}
        m = NAME_RE.match(req)
        if not m:
            continue
        raw_name, extras = m.group(1), (m.group(2) or "")
        key = norm(raw_name)
        if key not in changed:
            continue
        # Preserve any environment marker (the "; ..." tail) on the requirement.
        marker = req.partition(";")[2].strip()
        marker = f"; {marker}" if marker else ""
        print(f"{scope}\t{raw_name}{extras}>={changed[key]}{marker}")

project = pp.get("project", {})
emit("main", project.get("dependencies", []))
for extra, reqs in project.get("optional-dependencies", {}).items():
    emit(f"optional:{extra}", reqs)
for group, reqs in pp.get("dependency-groups", {}).items():
    emit(f"group:{group}", reqs)
PY
)

if [[ ${#PLAN[@]} -eq 0 ]]; then
  echo "==> No direct dependencies changed; pyproject.toml is already up to date."
else
  # Group specs by scope so each scope needs only a single `uv add` invocation.
  declare -A SCOPE_SPECS=()
  for line in "${PLAN[@]}"; do
    scope="${line%%$'\t'*}"
    spec="${line#*$'\t'}"
    SCOPE_SPECS["${scope}"]+="${spec}"$'\n'
  done

  for scope in "${!SCOPE_SPECS[@]}"; do
    mapfile -t specs < <(printf '%s' "${SCOPE_SPECS[${scope}]}")
    case "${scope}" in
      main)       flags=() ;;
      group:*)    flags=(--group "${scope#group:}") ;;
      optional:*) flags=(--optional "${scope#optional:}") ;;
    esac
    echo "==> uv add ${flags[*]} ${specs[*]}"
    # --no-sync: defer environment changes to the single sync below.
    uv add --no-sync "${flags[@]}" "${specs[@]}"
  done
fi

echo "==> uv sync --all-extras --all-packages --all-groups"
uv sync --all-extras --all-packages --all-groups

if [[ ${#PLAN[@]} -gt 0 ]]; then
  echo "==> Done. Updated direct dependency constraints:"
  printf '   %s\n' "${PLAN[@]#*$'\t'}"
else
  echo "==> Done."
fi
