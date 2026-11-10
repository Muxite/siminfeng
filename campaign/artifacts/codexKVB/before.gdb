set pagination off
set confirm off
set breakpoint pending on
break ggml_backend_sched_split_graph
commands
silent
printf "SPLIT graph capacity=%d nodes=%d leaves=%d scheduler_hash=%zu\n", graph->size, graph->n_nodes, graph->n_leafs, sched->hash_set.size
continue
end
break ggml_abort
commands
silent
bt 18
frame 1
info args
info locals
quit
end
run
