printf "__REPL_READY__\n" >&2
while read sentinel_id code_len; do
    code=$(head -c "$code_len")
    eval "$code" < /dev/null
    _ec=$?
    printf "__STDOUT_END_%s__\n" "$sentinel_id"
    printf "__SENTINEL_%s_%d__\n" "$sentinel_id" "$_ec" >&2
done
