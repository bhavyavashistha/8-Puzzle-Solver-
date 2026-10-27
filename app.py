import heapq
import os
import random
import time
import gradio as gr
import networkx as nx
import plotly.graph_objects as go

GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)


def get_neighbors(state):
    neighbors = []
    blank = state.index(0)
    row, col = divmod(blank, 3)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            idx = nr * 3 + nc
            nxt = list(state)
            nxt[blank], nxt[idx] = nxt[idx], nxt[blank]
            neighbors.append(tuple(nxt))
    return neighbors


def manhattan_distance(state, goal=GOAL_STATE):
    distance = 0
    for tile in range(1, 9):
        r1, c1 = divmod(state.index(tile), 3)
        r2, c2 = divmod(goal.index(tile), 3)
        distance += abs(r1-r2) + abs(c1-c2)
    return distance


def a_star(start, goal=GOAL_STATE):
    frontier = [(manhattan_distance(start, goal), 0, start)]
    parent = {}
    g_cost = {start: 0}
    closed = set()
    while frontier:
        _, current_g, current = heapq.heappop(frontier)
        if current in closed:
            continue
        if current == goal:
            return current, parent, current_g, len(closed) + 1
        closed.add(current)
        for neighbor in get_neighbors(current):
            tentative_g = current_g + 1
            if tentative_g < g_cost.get(neighbor, float('inf')):
                g_cost[neighbor] = tentative_g
                parent[neighbor] = current
                f = tentative_g + manhattan_distance(neighbor, goal)
                heapq.heappush(frontier, (f, tentative_g, neighbor))
    return None, parent, None, len(closed)


def reconstruct_path(parent, start, goal):
    if start == goal:
        return [start]
    if goal not in parent:
        return []
    path = [goal]
    while path[-1] != start:
        prev = parent.get(path[-1])
        if prev is None:
            return []
        path.append(prev)
    path.reverse()
    return path


def get_move(previous_state, current_state):
    old, new = previous_state.index(0), current_state.index(0)
    r1, c1 = divmod(old, 3)
    r2, c2 = divmod(new, 3)
    if r2 < r1: return 'UP'
    if r2 > r1: return 'DOWN'
    if c2 < c1: return 'LEFT'
    if c2 > c1: return 'RIGHT'
    return 'START'


def create_random_puzzle(goal_state=GOAL_STATE, min_moves=10, max_moves=30):
    current = goal_state
    previous = None
    for _ in range(random.randint(min_moves, max_moves)):
        choices = [s for s in get_neighbors(current) if s != previous]
        previous, current = current, random.choice(choices)
    return current


def puzzle_to_html(state, previous_state=None):
    moved_tile = None
    if previous_state is not None:
        changed = [i for i, (a, b) in enumerate(zip(previous_state, state)) if a != b]
        moved_tile = next((state[i] for i in changed if state[i] != 0), None)
    cells = []
    for value in state:
        if value == 0:
            cells.append('<div class="tile blank"></div>')
        else:
            extra = ' moved' if value == moved_tile else ''
            cells.append(f'<div class="tile{extra}">{value}</div>')
    return '<div class="board-wrap"><div class="board">' + ''.join(cells) + '</div></div>'


def step_info(path, index):
    if not path:
        return '<div class="info-card"><h3>Ready to solve</h3><p>Generate a puzzle and click Solve with A*.</p></div>'
    index = max(0, min(int(index), len(path)-1))
    if index == 0:
        move, message = 'START', 'This is the initial puzzle.'
    else:
        move = get_move(path[index-1], path[index])
        message = f'The blank moved {move.lower()}.'
    if index == len(path)-1:
        move, message = 'GOAL REACHED', 'A* has reached the goal state!'
    progress = int(index / max(1, len(path)-1) * 100)
    return f'''<div class="info-card"><span class="pill">STEP {index} / {len(path)-1}</span><h2>{move}</h2><p>{message}</p><div class="progress-track"><div class="progress-fill" style="width:{progress}%"></div></div><p class="muted">{progress}% of solution viewed</p></div>'''


def statistics_html(moves=None, nodes=None, elapsed=None):
    moves = '—' if moves is None else str(moves)
    nodes = '—' if nodes is None else str(nodes)
    elapsed = '—' if elapsed is None else f'{elapsed:.5f} seconds'
    return f'''<div class="stats"><div class="stat"><small>SOLUTION MOVES</small><h2>{moves}</h2></div><div class="stat"><small>NODES EXPLORED</small><h2>{nodes}</h2></div><div class="stat"><small>SEARCH TIME</small><h2>{elapsed}</h2></div></div>'''


def create_state_space_graph(solution_path):
    if not solution_path:
        return go.Figure()
    graph = nx.DiGraph()
    for i, state in enumerate(solution_path):
        g = i
        h = manhattan_distance(state)
        label = f"Step {i}<br>g={g}, h={h}, f={g+h}<br>" + ' '.join('_' if x == 0 else str(x) for x in state)
        graph.add_node(state, label=label, step=i)
        if i:
            graph.add_edge(solution_path[i-1], state)
    positions = {state: (i, 0) for i, state in enumerate(solution_path)}
    edge_x, edge_y = [], []
    for source, target in graph.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edges = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=2, color='#64748b'), hoverinfo='none')
    node_x, node_y, labels, colors, hover = [], [], [], [], []
    for state in solution_path:
        x, y = positions[state]
        node_x.append(x); node_y.append(y)
        labels.append(f"Step {graph.nodes[state]['step']}")
        colors.append('#16a34a' if state == GOAL_STATE else '#2563eb')
        hover.append(graph.nodes[state]['label'])
    nodes = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=labels, textposition='bottom center', hovertext=hover, hovertemplate='%{hovertext}<extra></extra>', marker=dict(size=35, color=colors, line=dict(width=2, color='white')))
    fig = go.Figure(data=[edges, nodes])
    fig.update_layout(title='A* State-Space Graph — Solution Path', showlegend=False, template='plotly_dark', height=400, margin=dict(l=20, r=20, t=60, b=60), xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def generate_new_puzzle():
    puzzle = create_random_puzzle()
    return puzzle, [], 0, puzzle_to_html(puzzle), step_info([], 0), statistics_html(), False, gr.update(value=0, maximum=1, interactive=False), go.Figure()


def solve_current_puzzle(puzzle):
    started = time.perf_counter()
    result, parent, moves, nodes = a_star(puzzle, GOAL_STATE)
    elapsed = time.perf_counter() - started
    if result is None:
        return puzzle, [], 0, puzzle_to_html(puzzle), '<div class="info-card">No solution found.</div>', statistics_html(nodes=nodes, elapsed=elapsed), False, gr.update(value=0, maximum=1, interactive=False), go.Figure()
    path = reconstruct_path(parent, puzzle, GOAL_STATE)
    return puzzle, path, 0, puzzle_to_html(path[0]), step_info(path, 0), statistics_html(moves=len(path)-1, nodes=nodes, elapsed=elapsed), False, gr.update(value=0, maximum=len(path)-1, interactive=True), create_state_space_graph(path)


def show_step(path, index):
    if not path:
        return 0, '<p>Solve a puzzle first.</p>', step_info([], 0), gr.update(value=0)
    index = max(0, min(int(index), len(path)-1))
    previous = path[index-1] if index > 0 else None
    return index, puzzle_to_html(path[index], previous), step_info(path, index), gr.update(value=index)


def next_step(path, index):
    return show_step(path, min(int(index)+1, len(path)-1)) if path else show_step(path, 0)


def previous_step(path, index):
    return show_step(path, max(int(index)-1, 0)) if path else show_step(path, 0)


def playback_tick(path, index, playing):
    if not path or not playing:
        return index, gr.update(), gr.update(), playing, gr.update()
    if index >= len(path)-1:
        return index, gr.update(), gr.update(), False, gr.update(value=index)
    new_index = index + 1
    previous = path[new_index-1] if new_index > 0 else None
    return new_index, puzzle_to_html(path[new_index], previous), step_info(path, new_index), new_index < len(path)-1, gr.update(value=new_index)


CUSTOM_CSS = '''
.gradio-container { max-width: 1100px !important; margin: auto !important; background: radial-gradient(ellipse at top left, #202b59, transparent 45%), #0b1020 !important; color: #edf2ff !important; }
#hero { padding: 28px; border-radius: 22px; background: linear-gradient(120deg, #25376b, #42276b); border: 1px solid #5966a5; margin-bottom: 16px; }
#hero h1 { color: white; font-size: 36px; margin-bottom: 8px; } #hero p { color: #d4ddff; }
.panel { background: rgba(21, 30, 56, .94); border: 1px solid #344466; border-radius: 20px; padding: 18px; }
.board-wrap { display: flex; justify-content: center; padding: 18px 4px; }
.board { display: grid; grid-template-columns: repeat(3, minmax(55px, 82px)); gap: 9px; padding: 12px; border-radius: 20px; background: #080e1f; border: 1px solid #344466; }
.tile { width: clamp(55px, 15vw, 82px); height: clamp(55px, 15vw, 82px); display: flex; justify-content: center; align-items: center; border-radius: 14px; color: #081225; font-size: 30px; font-weight: 800; background: linear-gradient(145deg, #60a5fa, #a78bfa); box-shadow: 0 5px 14px #0004; }
.tile.moved { background: linear-gradient(145deg, #5eead4, #38bdf8); } .tile.blank { background: #ffffff0a; border: 1px dashed #64748b77; box-shadow: none; }
.info-card { padding: 18px; border: 1px solid #344466; border-radius: 16px; background: linear-gradient(135deg, #172449, #17182f); } .info-card h2 { color: #c4b5fd; }
.pill { background: #3b82f633; color: #93c5fd; padding: 6px 10px; border-radius: 30px; font-size: 12px; font-weight: bold; }
.progress-track { height: 8px; background: #ffffff18; border-radius: 20px; overflow: hidden; } .progress-fill { height: 100%; background: linear-gradient(90deg, #38bdf8, #a78bfa); }
.muted { color: #a5b4d1; font-size: 12px; } .stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.stat { padding: 15px; border-radius: 14px; border: 1px solid #344466; background: #111b35; overflow-wrap: anywhere; } .stat small { color: #9eafd0; } .stat h2 { color: #e0e7ff; font-size: 21px; }
button.primary { background: linear-gradient(100deg, #2563eb, #7c3aed) !important; border: none !important; } footer { display: none !important; }
@media(max-width: 600px) { #hero { padding: 20px; } #hero h1 { font-size: 27px; } .stats { grid-template-columns: 1fr; } }
'''

initial_puzzle = create_random_puzzle()
with gr.Blocks(title='8-Puzzle Solver | A* Search', css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue='blue', secondary_hue='violet', neutral_hue='slate')) as app:
    gr.HTML('<div id="hero"><h1>🧩 8-Puzzle Solver</h1><p>Explore Artificial Intelligence through A* Search and Manhattan Distance.</p></div>')
    puzzle_state = gr.State(initial_puzzle)
    path_state = gr.State([])
    index_state = gr.State(0)
    playing_state = gr.State(False)
    with gr.Row():
        with gr.Column(elem_classes=['panel']):
            gr.Markdown('## 🧩 Puzzle Board')
            board_display = gr.HTML(puzzle_to_html(initial_puzzle))
            with gr.Row():
                new_button = gr.Button('🎲 New Puzzle')
                solve_button = gr.Button('✨ Solve with A*', variant='primary')
            gr.Markdown('## 🎮 Playback Controls')
            with gr.Row():
                play_button = gr.Button('▶ Play')
                pause_button = gr.Button('⏸ Pause')
            with gr.Row():
                previous_button = gr.Button('← Previous')
                next_button = gr.Button('Next →')
            step_slider = gr.Slider(minimum=0, maximum=1, value=0, step=1, label='Jump to a step', interactive=False)
        with gr.Column(elem_classes=['panel']):
            gr.Markdown('## 🧭 A* Move Inspector')
            step_display = gr.HTML(step_info([], 0))
            gr.Markdown('## 📊 Search Statistics')
            stats_display = gr.HTML(statistics_html())
            gr.Markdown('## 🌐 A* State-Space Graph')
            graph_display = gr.Plot(label='Solution State Transitions')
    timer = gr.Timer(0.8, active=True)
    new_button.click(generate_new_puzzle, inputs=[], outputs=[puzzle_state, path_state, index_state, board_display, step_display, stats_display, playing_state, step_slider, graph_display])
    solve_button.click(solve_current_puzzle, inputs=[puzzle_state], outputs=[puzzle_state, path_state, index_state, board_display, step_display, stats_display, playing_state, step_slider, graph_display])
    play_button.click(lambda: True, inputs=[], outputs=[playing_state])
    pause_button.click(lambda: False, inputs=[], outputs=[playing_state])
    next_button.click(next_step, inputs=[path_state, index_state], outputs=[index_state, board_display, step_display, step_slider])
    previous_button.click(previous_step, inputs=[path_state, index_state], outputs=[index_state, board_display, step_display, step_slider])
    step_slider.change(show_step, inputs=[path_state, step_slider], outputs=[index_state, board_display, step_display, step_slider])
    timer.tick(playback_tick, inputs=[path_state, index_state, playing_state], outputs=[index_state, board_display, step_display, playing_state, step_slider])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '10000'))
    app.launch(server_name='0.0.0.0', server_port=port)
