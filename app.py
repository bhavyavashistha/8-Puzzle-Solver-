import os
import heapq
import random
import time
import gradio as gr

GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)


def get_neighbors(state):
    neighbors = []
    blank = state.index(0)
    row, col = divmod(blank, 3)

    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_index = new_row * 3 + new_col
            new_state = list(state)
            new_state[blank], new_state[new_index] = new_state[new_index], new_state[blank]
            neighbors.append(tuple(new_state))
    return neighbors


def manhattan_distance(state, goal):
    distance = 0
    for tile in range(1, 9):
        current_index = state.index(tile)
        goal_index = goal.index(tile)
        current_row, current_col = divmod(current_index, 3)
        goal_row, goal_col = divmod(goal_index, 3)
        distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    return distance


def a_star(start, goal):
    open_list = [(manhattan_distance(start, goal), start)]
    g_cost = {start: 0}
    parent = {}
    closed_set = set()

    while open_list:
        _, current = heapq.heappop(open_list)
        if current in closed_set:
            continue
        if current == goal:
            return current, parent, g_cost[current], len(closed_set)

        closed_set.add(current)
        for neighbor in get_neighbors(current):
            new_g = g_cost[current] + 1
            if neighbor not in g_cost or new_g < g_cost[neighbor]:
                g_cost[neighbor] = new_g
                parent[neighbor] = current
                f_cost = new_g + manhattan_distance(neighbor, goal)
                heapq.heappush(open_list, (f_cost, neighbor))

    return None, parent, None, len(closed_set)


def reconstruct_path(parent, start, goal):
    path = []
    current = goal
    while current != start:
        if current not in parent:
            return []
        path.append(current)
        current = parent[current]
    path.append(start)
    path.reverse()
    return path


def get_move(previous_state, current_state):
    previous_blank = previous_state.index(0)
    current_blank = current_state.index(0)
    row1, col1 = divmod(previous_blank, 3)
    row2, col2 = divmod(current_blank, 3)
    if row2 < row1:
        return "UP"
    if row2 > row1:
        return "DOWN"
    if col2 < col1:
        return "LEFT"
    return "RIGHT"


def generate_random_puzzle(goal_state, min_moves=10, max_moves=30):
    current = goal_state
    previous = None
    for _ in range(random.randint(min_moves, max_moves)):
        choices = [state for state in get_neighbors(current) if state != previous]
        next_state = random.choice(choices)
        previous, current = current, next_state
    return current


def puzzle_to_html(state):
    cells = []
    for value in state:
        label = "" if value == 0 else str(value)
        background = "#dbeafe" if value else "#9ca3af"
        cells.append(f'''<div style="display:flex;justify-content:center;align-items:center;
            width:76px;height:76px;border:2px solid #334155;border-radius:8px;
            background:{background};color:#111827;font-size:28px;font-weight:bold;">{label}</div>''')
    return '<div style="display:grid;grid-template-columns:repeat(3,76px);gap:5px;justify-content:center;margin:18px auto;">' + ''.join(cells) + '</div>'


def render_solution_steps(solution_path):
    if not solution_path:
        return "<p>No solution path available.</p>"

    html = '<div style="max-height:750px;overflow-y:auto;padding:12px;font-family:Arial,sans-serif;"><h2 style="text-align:center;">A* Solution — Step by Step</h2>'
    for i, state in enumerate(solution_path):
        move_text = "Initial State" if i == 0 else get_move(solution_path[i - 1], state)
        html += f'''<div style="border:1px solid #888;border-radius:12px;padding:12px;margin:14px auto;max-width:420px;text-align:center;">
            <h3>Step {i}</h3><p><b>Move:</b> {move_text}</p>{puzzle_to_html(state)}</div>'''
    html += '<h2 style="text-align:center;color:green;">Goal Reached!</h2></div>'
    return html


current_puzzle = generate_random_puzzle(GOAL_STATE)


def generate_new_puzzle():
    global current_puzzle
    current_puzzle = generate_random_puzzle(GOAL_STATE)
    return (
        puzzle_to_html(current_puzzle),
        "<p>Click <b>Solve Using A*</b> to display every solution step.</p>",
        "<p style='text-align:center;color:green;'>New solvable puzzle generated.</p>"
    )


def solve_current_puzzle():
    start = current_puzzle
    start_time = time.perf_counter()
    result, parent, moves, nodes_explored = a_star(start, GOAL_STATE)
    elapsed = time.perf_counter() - start_time

    if result is None:
        return puzzle_to_html(start), "<p>No solution path available.</p>", "<p style='color:red;'>No solution found.</p>"

    path = reconstruct_path(parent, start, GOAL_STATE)
    status = f'''<div style="text-align:center;padding:15px;">
        <h2 style="color:green;">Goal Reached!</h2>
        <p><b>Algorithm:</b> A* Search</p>
        <p><b>Heuristic:</b> Manhattan Distance</p>
        <p><b>Solution moves:</b> {len(path) - 1}</p>
        <p><b>States displayed:</b> {len(path)}</p>
        <p><b>Nodes explored:</b> {nodes_explored}</p>
        <p><b>Execution time:</b> {elapsed:.6f} seconds</p>
        <p><b>Status:</b> Solved</p></div>'''
    return puzzle_to_html(start), render_solution_steps(path), status


with gr.Blocks(title="8-Puzzle Solver") as app:
    gr.Markdown("""# 🧩 8-Puzzle Solver Using A* Search
An AI puzzle solver using **A* Search** and the **Manhattan Distance heuristic**.""")
    gr.Markdown("### Initial Puzzle")
    puzzle_display = gr.HTML(value=puzzle_to_html(current_puzzle))
    with gr.Row():
        new_button = gr.Button("🎲 Generate New Puzzle")
        solve_button = gr.Button("▶ Solve Using A*", variant="primary")
    gr.Markdown("### 📊 Solution Statistics")
    status_display = gr.HTML("Generate a puzzle and solve it.")
    gr.Markdown("### 🪜 Every Step of the Solution")
    solution_display = gr.HTML("Your solution steps will appear here.")
    new_button.click(generate_new_puzzle, inputs=[], outputs=[puzzle_display, solution_display, status_display])
    solve_button.click(solve_current_puzzle, inputs=[], outputs=[puzzle_display, solution_display, status_display])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.launch(server_name="0.0.0.0", server_port=port)
