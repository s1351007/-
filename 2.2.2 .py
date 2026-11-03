import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ==================================================
# 1. 原始參數設定
# ==================================================

np.random.seed(7)

# 巢穴數量
n = 15

# 巢穴放棄比例
pa = 0.25

# 搜尋範圍
xmin = -5
xmax = 5

# 最大迭代次數
# 原始程式沒有設定迭代次數，因此新增此參數
max_iter = 50

# 動態更新速度，單位為毫秒
update_interval = 300


# ==================================================
# 2. 目標函數
# ==================================================

def objective(x):
    return (x - 2) ** 2


# ==================================================
# 3. 初始化候選解
# ==================================================

nests = np.random.uniform(xmin, xmax, n)

fitness = objective(nests)

# 初始最佳位置
best_index = np.argmin(fitness)

best_x = nests[best_index]

best_cost = fitness[best_index]


# ==================================================
# 4. 迭代相關變數
# ==================================================

iteration = 0

# 儲存歷史資料
history = []

# Tkinter 排程 ID
job = None

# 搜尋是否正在執行
running = False


# ==================================================
# 5. 建立 Tkinter 視窗
# ==================================================

root = tk.Tk()

root.title("Cuckoo Search Algorithm - 動態搜尋")

root.geometry("1200x700")


# ==================================================
# 6. 建立主畫面
# ==================================================

main_frame = ttk.Frame(root)

main_frame.pack(
    fill=tk.BOTH,
    expand=True,
    padx=10,
    pady=10
)


# 左側：目標函數圖形
left_frame = ttk.Frame(main_frame)

left_frame.pack(
    side=tk.LEFT,
    fill=tk.BOTH,
    expand=True
)


# 右側：資訊與表格
right_frame = ttk.Frame(main_frame)

right_frame.pack(
    side=tk.RIGHT,
    fill=tk.BOTH,
    expand=True,
    padx=10
)


# ==================================================
# 7. 左側圖形設定
# ==================================================

fig, ax = plt.subplots(figsize=(7, 5))

# 目標函數曲線
x_values = np.linspace(xmin, xmax, 300)

y_values = objective(x_values)

ax.plot(
    x_values,
    y_values,
    label="Objective Function",
    linewidth=2
)

# 最佳位置
best_point, = ax.plot(
    [best_x],
    [best_cost],
    "ro",
    markersize=8,
    label="Best Solution"
)

# 候選解
candidate_points, = ax.plot(
    nests,
    fitness,
    "bo",
    markersize=5,
    label="Candidate Solutions"
)

# 設定圖形
ax.set_title("Cuckoo Search Algorithm")

ax.set_xlabel("x")

ax.set_ylabel("f(x)")

ax.set_xlim(xmin, xmax)

ax.set_ylim(0, 50)

ax.grid(True)

ax.legend()

fig.tight_layout()


# 將 Matplotlib 放入 Tkinter
canvas = FigureCanvasTkAgg(
    fig,
    master=left_frame
)

canvas_widget = canvas.get_tk_widget()

canvas_widget.pack(
    fill=tk.BOTH,
    expand=True
)


# ==================================================
# 8. 右側資訊區
# ==================================================

info_frame = ttk.LabelFrame(
    right_frame,
    text="搜尋資訊"
)

info_frame.pack(
    fill=tk.X,
    pady=5
)


# 迭代次數
iteration_label = ttk.Label(
    info_frame,
    text="迭代次數：0 / 100"
)

iteration_label.pack(
    anchor=tk.W,
    padx=10,
    pady=5
)


# 最佳位置
best_x_label = ttk.Label(
    info_frame,
    text=f"最佳位置：{best_x:.6f}"
)

best_x_label.pack(
    anchor=tk.W,
    padx=10,
    pady=5
)


# 最佳函數值
best_cost_label = ttk.Label(
    info_frame,
    text=f"最佳函數值：{best_cost:.6f}"
)

best_cost_label.pack(
    anchor=tk.W,
    padx=10,
    pady=5
)


# 誤差
error_label = ttk.Label(
    info_frame,
    text=f"誤差：{best_cost:.6f}"
)

error_label.pack(
    anchor=tk.W,
    padx=10,
    pady=5
)


# 搜尋狀態
status_label = ttk.Label(
    info_frame,
    text="搜尋尚未開始"
)

status_label.pack(
    anchor=tk.W,
    padx=10,
    pady=5
)


# ==================================================
# 9. 建立資料表格
# ==================================================

table_frame = ttk.LabelFrame(
    right_frame,
    text="迭代紀錄"
)

table_frame.pack(
    fill=tk.BOTH,
    expand=True,
    pady=5
)


# 表格欄位
columns = (
    "iteration",
    "best_x",
    "best_cost",
    "error"
)


tree = ttk.Treeview(
    table_frame,
    columns=columns,
    show="headings",
    height=20
)


# 欄位標題
tree.heading(
    "iteration",
    text="迭代次數"
)

tree.heading(
    "best_x",
    text="最佳位置"
)

tree.heading(
    "best_cost",
    text="最佳函數值"
)

tree.heading(
    "error",
    text="誤差"
)


# 欄位寬度
tree.column(
    "iteration",
    width=80,
    anchor=tk.CENTER
)

tree.column(
    "best_x",
    width=100,
    anchor=tk.CENTER
)

tree.column(
    "best_cost",
    width=120,
    anchor=tk.CENTER
)

tree.column(
    "error",
    width=100,
    anchor=tk.CENTER
)


# 垂直捲動條
scrollbar = ttk.Scrollbar(
    table_frame,
    orient=tk.VERTICAL,
    command=tree.yview
)

tree.configure(
    yscrollcommand=scrollbar.set
)


scrollbar.pack(
    side=tk.RIGHT,
    fill=tk.Y
)

tree.pack(
    side=tk.LEFT,
    fill=tk.BOTH,
    expand=True
)


# ==================================================
# 10. 更新圖形
# ==================================================

def update_plot():

    # 更新候選解
    candidate_points.set_data(
        nests,
        fitness
    )

    # 更新最佳解
    best_point.set_data(
        [best_x],
        [best_cost]
    )

    # 更新標題
    ax.set_title(
        f"Cuckoo Search - Iteration {iteration}"
    )

    # 更新圖形
    canvas.draw_idle()


# ==================================================
# 11. 更新資訊
# ==================================================

def update_info():

    # 迭代次數
    iteration_label.config(
        text=f"迭代次數：{iteration} / {max_iter}"
    )

    # 最佳位置
    best_x_label.config(
        text=f"最佳位置：{best_x:.6f}"
    )

    # 最佳函數值
    best_cost_label.config(
        text=f"最佳函數值：{best_cost:.6f}"
    )

    # 誤差
    error_label.config(
        text=f"誤差：{best_cost:.6f}"
    )


# ==================================================
# 12. 更新表格
# ==================================================

def update_table():

    # 記錄目前迭代資料
    tree.insert(
        "",
        tk.END,
        values=(
            iteration,
            f"{best_x:.6f}",
            f"{best_cost:.6f}",
            f"{best_cost:.6f}"
        )
    )

    # 自動移動至最新紀錄
    tree.yview_moveto(1)


# ==================================================
# 13. Cuckoo Search 更新
# ==================================================

def update_search():

    global nests
    global fitness

    global best_x
    global best_cost

    global iteration
    global running
    global job

    # 搜尋已停止
    if not running:
        return

    # ----------------------------------------------
    # Step 1：產生新候選解
    # ----------------------------------------------

    new_nests = nests + np.random.normal(
        0,
        1,
        n
    )

    # 限制搜尋範圍
    new_nests = np.clip(
        new_nests,
        xmin,
        xmax
    )

    # 計算新候選解的函數值
    new_fitness = objective(new_nests)


    # ----------------------------------------------
    # Step 2：比較新舊候選解
    # ----------------------------------------------

    for i in range(n):

        if new_fitness[i] < fitness[i]:

            nests[i] = new_nests[i]

            fitness[i] = new_fitness[i]


    # ----------------------------------------------
    # Step 3：放棄部分巢穴
    # ----------------------------------------------

    count = max(
        1,
        int(pa * n)
    )

    # 找出函數值較差的巢穴
    worst_indices = np.argsort(fitness)[-count:]

    # 重新產生候選解
    nests[worst_indices] = np.random.uniform(
        xmin,
        xmax,
        count
    )

    # 重新計算函數值
    fitness[worst_indices] = objective(
        nests[worst_indices]
    )


    # ----------------------------------------------
    # Step 4：更新最佳解
    # ----------------------------------------------

    current_best_index = np.argmin(fitness)

    current_best_x = nests[current_best_index]

    current_best_cost = fitness[current_best_index]


    # 如果找到更好的解，更新最佳解
    if current_best_cost < best_cost:

        best_x = current_best_x

        best_cost = current_best_cost


    # ----------------------------------------------
    # Step 5：更新迭代次數
    # ----------------------------------------------

    iteration += 1


    # ----------------------------------------------
    # Step 6：更新圖形與資訊
    # ----------------------------------------------

    update_plot()

    update_info()

    update_table()


    # ----------------------------------------------
    # Step 7：固定迭代次數停止
    # ----------------------------------------------

    if iteration >= max_iter:

        running = False

        job = None

        status_label.config(
            text=f"搜尋完成：共執行 {max_iter} 次迭代"
        )

        start_button.config(
            state=tk.NORMAL
        )

        stop_button.config(
            state=tk.DISABLED
        )

        return


    # ----------------------------------------------
    # Step 8：繼續下一次迭代
    # ----------------------------------------------

    job = root.after(
        update_interval,
        update_search
    )


# ==================================================
# 14. 開始搜尋
# ==================================================

def start_search():

    global running
    global job

    # 避免重複開始
    if running:
        return

    running = True

    status_label.config(
        text="搜尋進行中..."
    )

    start_button.config(
        state=tk.DISABLED
    )

    stop_button.config(
        state=tk.NORMAL
    )

    # 開始更新
    update_search()


# ==================================================
# 15. 停止搜尋
# ==================================================

def stop_search():

    global running
    global job

    running = False

    # 取消尚未執行的排程
    if job is not None:

        root.after_cancel(job)

        job = None

    status_label.config(
        text=f"搜尋已停止：目前迭代 {iteration}"
    )

    start_button.config(
        state=tk.NORMAL
    )

    stop_button.config(
        state=tk.DISABLED
    )


# ==================================================
# 16. 按鈕區
# ==================================================

button_frame = ttk.Frame(
    right_frame
)

button_frame.pack(
    fill=tk.X,
    pady=10
)


# 開始按鈕
start_button = ttk.Button(
    button_frame,
    text="開始搜尋",
    command=start_search
)

start_button.pack(
    side=tk.LEFT,
    padx=5
)


# 停止按鈕
stop_button = ttk.Button(
    button_frame,
    text="停止搜尋",
    command=stop_search,
    state=tk.DISABLED
)

stop_button.pack(
    side=tk.LEFT,
    padx=5
)


# ==================================================
# 17. 執行 Tkinter
# ==================================================

root.mainloop()