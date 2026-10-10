#!/bin/bash
# Build the project image of each example, run it, POST its sample_request.json and check HTTP 200 and the
# figures the example's README shows. Podman by default (CONTAINER_CLI=docker also works).
#
#   docker/smoke.sh [--prune] [--cpu host] [example ...]      (no example: all of them)
#
# Env: RUN_WRAP prefixes each example's build and run (for example a lock and a memory-capped scope), MEMORY
#      (container limit, default 2g), WORKERS (default 1), PORT (default 18080).
cd "$(dirname "$0")/.." || exit 1
cli=${CONTAINER_CLI:-podman}
prune=0 cpu=generic one=0
while [[ $1 == --* ]]; do
    case $1 in --prune) prune=1 ;; --one) one=1 ;; --cpu) cpu=$2; shift ;; *) echo "unknown option $1"; exit 2 ;; esac
    shift
done
examples=("$@")
[[ ${#examples[@]} -gt 0 ]] || examples=($(cd examples && ls -d */ | tr -d /))
port=${PORT:-18080}

# The first JSON block after the curl command in the README is the documented response: compare every
# scalar it shows that the response has.
check() {  # readme response-file
    python3 - "$1" "$2" <<'PY'
import json, sys
lines = open(sys.argv[1]).read().splitlines()
start = next(i for i, l in enumerate(lines) if "curl" in l)
i = next(i for i in range(start, len(lines)) if lines[i].startswith("```json")) + 1
j = next(j for j in range(i, len(lines)) if lines[j].startswith("```"))
want, got = json.loads("\n".join(lines[i:j])), json.load(open(sys.argv[2]))
# keys the README computes with jq (a list's length, say) are not in the response
scalars = {k: v for k, v in want.items() if not isinstance(v, (list, dict)) and k in got}
bad = {k: (v, got.get(k)) for k, v in scalars.items()
       if got.get(k) != v and not (isinstance(v, float) and abs(got.get(k, 1e99) - v) < 1e-6)}
print(f"{len(scalars) - len(bad)}/{len(scalars)} figures match" + (f" (differ: {bad})" if bad else ""))
sys.exit(1 if bad or not scalars else 0)
PY
}

now() { date +%s.%N; }
one() {
    local fail=0
    ex=$1
    img=smoke-$ex:test name=smoke-$ex
    pipeline=$(cd examples/$ex && ls -d */pipeline.py | head -1 | sed 's|/pipeline.py|.pipeline:build|')
    t0=$(now)
    $cli build -q --target project --build-arg PROJECT=examples/$ex --build-arg PIPELINE=$pipeline \
        --build-arg CPU_TARGET=$cpu -t $img -f docker/Dockerfile . > /dev/null 2> ${TMPDIR:-.}/build-$ex.log \
        || { printf '%-26s build failed, see %s/build-%s.log\n' $ex ${TMPDIR:-.} $ex; return 1; }
    rm -f ${TMPDIR:-.}/build-$ex.log
    t1=$(now)
    size=$($cli image inspect $img --format '{{.Size}}' | awk '{printf "%dMB", $1/1e6}')
    $cli run -d --name $name --memory=${MEMORY:-2g} -p 127.0.0.1:$port:8080 \
        -e DECIDER_SERVE__WORKERS=${WORKERS:-1} $img > /dev/null || return 1
    t2=$(now) ready=
    for _ in $(seq 1 3000); do
        [[ $(curl -s -o /dev/null -w '%{http_code}' localhost:$port/ping) == 200 ]] && { ready=$(now); break; }
        sleep 0.1
    done
    if [[ -z $ready ]]; then
        printf '%-26s never ready\n' $ex; $cli logs $name 2>&1 | tail -5; fail=1
    else
        out=$(mktemp -p "${TMPDIR:-.}")
        read -r http secs < <(curl -s -o $out -w '%{http_code} %{time_total}' --data-binary @examples/$ex/sample_request.json \
            -H 'content-type: application/json' localhost:$port/invocations)
        figures=$(check examples/$ex/README.md $out) || fail=1
        [[ $http == 200 ]] || fail=1
        printf '%-26s %8.0f %9s %9.1f %11.1f %5s  %s\n' $ex "$(echo "$t1 - $t0" | bc)" $size \
            "$(echo "$ready - $t2" | bc)" "$(echo "$secs * 1000" | bc)" $http "$figures"
        rm -f $out
    fi
    $cli rm -f $name > /dev/null
    [[ $prune == 0 ]] || $cli rmi $img > /dev/null
    return $fail
}

if [[ $one == 1 ]]; then one "${examples[0]}"; exit; fi
printf '%-26s %8s %9s %9s %11s %5s  %s\n' example build_s size ready_s first_ms http figures
fail=0
for ex in "${examples[@]}"; do
    $RUN_WRAP "$0" --one --cpu $cpu $([[ $prune == 1 ]] && echo --prune) $ex || fail=1
done
exit $fail
