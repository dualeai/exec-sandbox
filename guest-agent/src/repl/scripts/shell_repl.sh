printf "__REPL_READY__\n" >&2
# reseed: 1 to give $RANDOM a new seed from the kernel ($SRANDOM) before the code runs
while read sentinel_id code_len reseed; do
    # code_len bytes, with no process: a builtin read; LC_ALL=C makes -N count bytes (user code
    # may set a UTF-8 locale)
    LC_ALL=C IFS= read -r -N "$code_len" code
    if [ "$reseed" = 1 ]; then RANDOM=$SRANDOM; fi  # an if: the code starts with $? = 0
    eval "$code" < /dev/null
    _ec=$?
    printf "__STDOUT_END_%s__\n" "$sentinel_id"
    printf "__SENTINEL_%s_%d__\n" "$sentinel_id" "$_ec" >&2
done
