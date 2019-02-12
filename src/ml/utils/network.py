def all_simple_paths_graph(graph, source, cutoff):
    if cutoff < 1:
        return
    visited = [source]
    stack = [iter(graph[source])]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.pop()
        elif len(visited) < cutoff:
            if child not in visited:
                visited.append(child)
                stack.append(iter(graph[child]))
        elif len(visited) == cutoff:
            yield visited[:]
            stack.pop()
            visited.pop()
