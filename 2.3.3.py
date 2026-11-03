
# ============================================================
# 2.3.3 Lévy Flight Distribution Analysis
#
# 主題：
# Lévy Flight 分布分析、Normal Distribution 比較
# 以及 Tkinter 動態搜尋模擬
#
# 使用環境：
# Spyder / Python
#
# 圖 2-5：PDF 機率密度比較
# 圖 2-6：Lévy Flight 局部分布直方圖
# 圖 2-7：CDF 累積分布比較
# 圖 2-8：CCDF 重尾分析
#
# 動畫 2-1：Lévy Flight 動態搜尋
#
# 本版本：
# 1. 使用 class 管理程式
# 2. 不使用 global
# 3. 修正動畫開始、暫停、重設
# 4. 加入速度控制
# 5. 加入 PNG 與 GIF 儲存
# ============================================================


import tkinter as tk

from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.figure import Figure

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from scipy.special import gamma

from scipy.stats import norm

from PIL import Image

from matplotlib.backends.backend_agg import FigureCanvasAgg


# ============================================================
# 第一部分：Lévy Flight 數學模型
# ============================================================

def levy_flight(n, lam=1.5, rng=None):
    """
    主題：
    使用 Mantegna 方法產生 Lévy Flight 隨機步長。

    數學公式：

    s = u / |v|^(1/lambda)

    u ~ Normal(0, sigma_u^2)
    v ~ Normal(0, 1)

    參數：
    n   = 需要產生的步長數量
    lam = Lévy 指數
    rng = 隨機數產生器

    回傳：
    n 個 Lévy Flight 隨機步長
    """

    if rng is None:
        rng = np.random.default_rng()

    if not 0 < lam < 2:
        raise ValueError("Lévy 指數 lambda 必須介於 0 和 2 之間。")

    # --------------------------------------------------------
    # 計算 sigma_u
    # --------------------------------------------------------

    sigma_u = (

        gamma(1 + lam)

        * np.sin(np.pi * lam / 2)

        /

        (
            gamma((1 + lam) / 2)

            * lam

            * 2 ** ((lam - 1) / 2)
        )

    ) ** (1 / lam)

    # --------------------------------------------------------
    # 產生常態隨機變數
    # --------------------------------------------------------

    u = rng.normal(
        loc=0,
        scale=sigma_u,
        size=n
    )

    v = rng.normal(
        loc=0,
        scale=1,
        size=n
    )

    # 避免分母接近零

    v = np.where(
        np.abs(v) < 1e-12,
        1e-12,
        v
    )

    # --------------------------------------------------------
    # Lévy Flight 步長
    # --------------------------------------------------------

    steps = (

        u

        /

        np.abs(v) ** (1 / lam)

    )

    return steps


# ============================================================
# 第二部分：Tkinter 主程式
# ============================================================

class LevyFlightApp:

    def __init__(self, root):

        # ----------------------------------------------------
        # 1. 主視窗
        # ----------------------------------------------------

        self.root = root

        self.root.title(
            "2.3.3 Lévy Flight Distribution Analysis"
        )

        self.root.geometry(
            "1200x850"
        )

        self.root.minsize(
            950,
            700
        )

        # ----------------------------------------------------
        # 2. 基本參數
        # ----------------------------------------------------

        self.sample_count = 10000

        self.levy_lambda = 1.5

        self.animation_steps_count = 300

        self.local_range = 10

        self.random_seed = 42

        # ----------------------------------------------------
        # 3. 隨機數產生器
        # ----------------------------------------------------

        self.rng = np.random.default_rng(
            self.random_seed
        )

        # ----------------------------------------------------
        # 4. 動畫控制變數
        # ----------------------------------------------------

        self.current_frame = 0

        self.animation_running = False

        self.animation_job = None

        self.animation_delay = 50

        # ----------------------------------------------------
        # 5. 產生分析資料
        # ----------------------------------------------------

        self.generate_data()

        # ----------------------------------------------------
        # 6. 建立介面
        # ----------------------------------------------------

        self.create_interface()

        # ----------------------------------------------------
        # 7. 建立分析圖
        # ----------------------------------------------------

        self.create_analysis_figure()

        # ----------------------------------------------------
        # 8. 建立動畫圖
        # ----------------------------------------------------

        self.create_animation_figure()

        # ----------------------------------------------------
        # 9. 關閉視窗事件
        # ----------------------------------------------------

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_application
        )

    # ========================================================
    # 1. 產生資料
    # ========================================================

    def generate_data(self):

        """
        主題：
        產生 Lévy Flight 與 Normal Distribution 樣本。

        使用相同樣本數進行比較。
        """

        # Lévy Flight

        self.levy_samples = levy_flight(

            self.sample_count,

            self.levy_lambda,

            self.rng

        )

        # Normal Distribution

        self.normal_samples = self.rng.normal(

            loc=0,

            scale=1,

            size=self.sample_count

        )

        # 動畫步長

        self.animation_steps = levy_flight(

            self.animation_steps_count,

            self.levy_lambda,

            self.rng

        )

        # 動畫累積位置

        self.animation_positions = np.concatenate(

            [

                [0],

                np.cumsum(self.animation_steps)

            ]

        )

        # 動畫時間

        self.animation_time = np.arange(

            self.animation_steps_count + 1

        )

    # ========================================================
    # 2. 建立 Tkinter 介面
    # ========================================================

    def create_interface(self):

        """
        主題：
        建立 Tkinter 分頁與控制介面。
        """

        # Notebook

        self.notebook = ttk.Notebook(
            self.root
        )

        self.notebook.pack(

            fill="both",

            expand=True

        )

        # ----------------------------------------------------
        # 分布分析分頁
        # ----------------------------------------------------

        self.analysis_tab = ttk.Frame(
            self.notebook
        )

        self.notebook.add(

            self.analysis_tab,

            text="Distribution Analysis"

        )

        # ----------------------------------------------------
        # 動態動畫分頁
        # ----------------------------------------------------

        self.animation_tab = ttk.Frame(
            self.notebook
        )

        self.notebook.add(

            self.animation_tab,

            text="Dynamic Lévy Flight"

        )

        # ----------------------------------------------------
        # 動畫控制面板
        # ----------------------------------------------------

        self.control_frame = ttk.Frame(
            self.animation_tab
        )

        self.control_frame.pack(

            fill="x",

            padx=10,

            pady=10

        )

        # ----------------------------------------------------
        # Start
        # ----------------------------------------------------

        self.start_button = ttk.Button(

            self.control_frame,

            text="Start",

            command=self.start_animation

        )

        self.start_button.pack(

            side="left",

            padx=5

        )

        # ----------------------------------------------------
        # Pause
        # ----------------------------------------------------

        self.pause_button = ttk.Button(

            self.control_frame,

            text="Pause",

            command=self.pause_animation

        )

        self.pause_button.pack(

            side="left",

            padx=5

        )

        # ----------------------------------------------------
        # Reset
        # ----------------------------------------------------

        self.reset_button = ttk.Button(

            self.control_frame,

            text="Reset",

            command=self.reset_animation

        )

        self.reset_button.pack(

            side="left",

            padx=5

        )

        # ----------------------------------------------------
        # Save PNG
        # ----------------------------------------------------

        self.save_png_button = ttk.Button(

            self.control_frame,

            text="Save PNG",

            command=self.save_analysis_png

        )

        self.save_png_button.pack(

            side="left",

            padx=5

        )

        # ----------------------------------------------------
        # Save GIF
        # ----------------------------------------------------

        self.save_gif_button = ttk.Button(

            self.control_frame,

            text="Save GIF",

            command=self.save_animation_gif

        )

        self.save_gif_button.pack(

            side="left",

            padx=5

        )

        # ----------------------------------------------------
        # 動畫速度
        # ----------------------------------------------------

        self.speed_frame = ttk.Frame(
            self.animation_tab
        )

        self.speed_frame.pack(

            fill="x",

            padx=10,

            pady=5

        )

        self.speed_label = ttk.Label(

            self.speed_frame,

            text="Animation Speed (ms):"

        )

        self.speed_label.pack(

            side="left",

            padx=5

        )

        self.speed_var = tk.IntVar(
            value=50
        )

        self.speed_slider = ttk.Scale(

            self.speed_frame,

            from_=10,

            to=200,

            variable=self.speed_var,

            orient="horizontal",

            command=self.change_speed

        )

        self.speed_slider.pack(

            side="left",

            fill="x",

            expand=True,

            padx=10

        )

        # ----------------------------------------------------
        # 狀態列
        # ----------------------------------------------------

        self.status_label = ttk.Label(

            self.animation_tab,

            text="Ready",

            anchor="center"

        )

        self.status_label.pack(

            fill="x",

            pady=5

        )

    # ========================================================
    # 3. 建立分析圖
    # ========================================================

    def create_analysis_figure(self):

        """
        主題：
        建立四張 Lévy Flight 分布分析圖。
        """

        self.analysis_fig = Figure(

            figsize=(12, 8),

            dpi=100

        )

        # 四張圖

        self.ax1 = self.analysis_fig.add_subplot(
            221
        )

        self.ax2 = self.analysis_fig.add_subplot(
            222
        )

        self.ax3 = self.analysis_fig.add_subplot(
            223
        )

        self.ax4 = self.analysis_fig.add_subplot(
            224
        )

        # ----------------------------------------------------
        # Figure 2-5
        # PDF 比較
        # ----------------------------------------------------

        local_levy = self.levy_samples[

            np.abs(self.levy_samples)

            <=

            self.local_range

        ]

        x_pdf = np.linspace(

            -self.local_range,

            self.local_range,

            1000

        )

        self.ax1.hist(

            local_levy,

            bins=100,

            range=(

                -self.local_range,

                self.local_range

            ),

            density=True,

            alpha=0.6,

            label="Levy Flight"

        )

        self.ax1.plot(

            x_pdf,

            norm.pdf(x_pdf),

            linewidth=2,

            label="Normal Distribution"

        )

        self.ax1.set_title(
            "Figure 2-5: PDF Comparison"
        )

        self.ax1.set_xlabel(
            "Step Size"
        )

        self.ax1.set_ylabel(
            "Probability Density"
        )

        self.ax1.legend()

        self.ax1.grid(
            True,
            alpha=0.3
        )

        # ----------------------------------------------------
        # Figure 2-6
        # 局部直方圖
        # ----------------------------------------------------

        self.ax2.hist(

            self.levy_samples,

            bins=100,

            range=(

                -self.local_range,

                self.local_range

            ),

            density=True,

            alpha=0.7

        )

        self.ax2.set_title(
            "Figure 2-6: Levy Local Distribution"
        )

        self.ax2.set_xlabel(
            "Step Size"
        )

        self.ax2.set_ylabel(
            "Probability Density"
        )

        self.ax2.grid(
            True,
            alpha=0.3
        )

        # ----------------------------------------------------
        # Figure 2-7
        # CDF
        # ----------------------------------------------------

        levy_sorted = np.sort(
            self.levy_samples
        )

        normal_sorted = np.sort(
            self.normal_samples
        )

        probability = (

            np.arange(

                1,

                self.sample_count + 1

            )

            /

            self.sample_count

        )

        self.ax3.plot(

            levy_sorted,

            probability,

            label="Levy Flight"

        )

        self.ax3.plot(

            normal_sorted,

            probability,

            label="Normal Distribution"

        )

        self.ax3.set_xlim(

            -self.local_range,

            self.local_range

        )

        self.ax3.set_title(
            "Figure 2-7: CDF Comparison"
        )

        self.ax3.set_xlabel(
            "Step Size"
        )

        self.ax3.set_ylabel(
            "Cumulative Probability"
        )

        self.ax3.legend()

        self.ax3.grid(
            True,
            alpha=0.3
        )

        # ----------------------------------------------------
        # Figure 2-8
        # CCDF
        # ----------------------------------------------------

        levy_abs = np.sort(

            np.abs(

                self.levy_samples

            )

        )

        normal_abs = np.sort(

            np.abs(

                self.normal_samples

            )

        )

        # Empirical survival probability
        # 避免機率為零

        ccdf = (

            self.sample_count

            -

            np.arange(

                self.sample_count

            )

        ) / self.sample_count

        ccdf = np.maximum(

            ccdf,

            1 / self.sample_count

        )

        self.ax4.loglog(

            levy_abs,

            ccdf,

            label="Levy Flight"

        )

        self.ax4.loglog(

            normal_abs,

            ccdf,

            label="Normal Distribution"

        )

        self.ax4.set_title(
            "Figure 2-8: CCDF Tail Comparison"
        )

        self.ax4.set_xlabel(
            "Absolute Step Size"
        )

        self.ax4.set_ylabel(
            "P(|X| > x)"
        )

        self.ax4.legend()

        self.ax4.grid(

            True,

            which="both",

            alpha=0.3

        )

        # ----------------------------------------------------
        # 排版
        # ----------------------------------------------------

        self.analysis_fig.tight_layout()

        # ----------------------------------------------------
        # 嵌入 Tkinter
        # ----------------------------------------------------

        self.analysis_canvas = FigureCanvasTkAgg(

            self.analysis_fig,

            master=self.analysis_tab

        )

        self.analysis_canvas.draw()

        self.analysis_canvas.get_tk_widget().pack(

            fill="both",

            expand=True

        )

        # ----------------------------------------------------
        # 工具列
        # ----------------------------------------------------

        toolbar_frame = ttk.Frame(
            self.analysis_tab
        )

        toolbar_frame.pack(
            fill="x"
        )

        self.analysis_toolbar = NavigationToolbar2Tk(

            self.analysis_canvas,

            toolbar_frame

        )

        self.analysis_toolbar.update()

    # ========================================================
    # 4. 建立動畫圖
    # ========================================================

    def create_animation_figure(self):

        """
        主題：
        建立動態搜尋位置與步長圖。
        """

        self.animation_fig = Figure(

            figsize=(10, 6),

            dpi=100

        )

        # 上方搜尋位置

        self.ax_position = self.animation_fig.add_subplot(
            211
        )

        # 下方步長

        self.ax_step = self.animation_fig.add_subplot(
            212
        )

        # ----------------------------------------------------
        # 搜尋軌跡
        # ----------------------------------------------------

        self.position_line, = self.ax_position.plot(

            [],

            [],

            linewidth=1.5,

            label="Search Trajectory"

        )

        # ----------------------------------------------------
        # 當前搜尋位置
        # ----------------------------------------------------

        self.position_point, = self.ax_position.plot(

            [],

            [],

            marker="o",

            markersize=8,

            label="Current Position"

        )

        # ----------------------------------------------------
        # 起始位置
        # ----------------------------------------------------

        self.start_point, = self.ax_position.plot(

            [0],

            [0],

            marker="s",

            markersize=8,

            label="Start"

        )

        # ----------------------------------------------------
        # 步長曲線
        # ----------------------------------------------------

        self.step_line, = self.ax_step.plot(

            [],

            [],

            linewidth=1

        )

        # ----------------------------------------------------
        # 搜尋位置軸
        # ----------------------------------------------------

        position_min = np.min(

            self.animation_positions

        )

        position_max = np.max(

            self.animation_positions

        )

        position_margin = max(

            1,

            0.1 * (

                position_max - position_min

            )

        )

        self.ax_position.set_xlim(

            0,

            self.animation_steps_count

        )

        self.ax_position.set_ylim(

            position_min - position_margin,

            position_max + position_margin

        )

        self.ax_position.set_xlabel(
            "Iteration"
        )

        self.ax_position.set_ylabel(
            "Search Position"
        )

        self.ax_position.set_title(
            "Lévy Flight Search Trajectory"
        )

        self.ax_position.legend(
            loc="upper left"
        )

        self.ax_position.grid(
            True,
            alpha=0.3
        )

        # ----------------------------------------------------
        # 步長軸
        # ----------------------------------------------------

        step_min = np.min(
            self.animation_steps
        )

        step_max = np.max(
            self.animation_steps
        )

        step_margin = max(

            1,

            0.1 * (

                step_max - step_min

            )

        )

        self.ax_step.set_xlim(

            0,

            self.animation_steps_count

        )

        self.ax_step.set_ylim(

            step_min - step_margin,

            step_max + step_margin

        )

        self.ax_step.set_xlabel(
            "Iteration"
        )

        self.ax_step.set_ylabel(
            "Step Size"
        )

        self.ax_step.set_title(
            "Lévy Flight Step Size"
        )

        self.ax_step.grid(
            True,
            alpha=0.3
        )

        # ----------------------------------------------------
        # 排版
        # ----------------------------------------------------

        self.animation_fig.tight_layout()

        # ----------------------------------------------------
        # 嵌入 Tkinter
        # ----------------------------------------------------

        self.animation_canvas = FigureCanvasTkAgg(

            self.animation_fig,

            master=self.animation_tab

        )

        self.animation_canvas.draw()

        self.animation_canvas.get_tk_widget().pack(

            fill="both",

            expand=True

        )

    # ========================================================
    # 5. 更新動畫
    # ========================================================

    def update_animation(self):

        """
        主題：
        更新 Lévy Flight 搜尋位置。

        不使用 global。
        """

        # ----------------------------------------------------
        # 如果動畫已停止
        # ----------------------------------------------------

        if not self.animation_running:

            return

        # ----------------------------------------------------
        # 如果已經完成全部迭代
        # ----------------------------------------------------

        if self.current_frame >= self.animation_steps_count:

            self.pause_animation()

            self.status_label.config(

                text="Animation Complete"

            )

            return

        # ----------------------------------------------------
        # 目前迭代
        # ----------------------------------------------------

        frame = self.current_frame

        # ----------------------------------------------------
        # 搜尋位置資料
        # ----------------------------------------------------

        x_data = self.animation_time[

            :frame + 2

        ]

        y_data = self.animation_positions[

            :frame + 2

        ]

        # ----------------------------------------------------
        # 更新搜尋軌跡
        # ----------------------------------------------------

        self.position_line.set_data(

            x_data,

            y_data

        )

        # ----------------------------------------------------
        # 更新目前位置
        # ----------------------------------------------------

        self.position_point.set_data(

            [self.animation_positions[frame + 1]],

            [self.animation_positions[frame + 1]]

        )

        # ----------------------------------------------------
        # 更新步長
        # ----------------------------------------------------

        step_x = np.arange(

            frame + 1

        )

        step_y = self.animation_steps[

            :frame + 1

        ]

        self.step_line.set_data(

            step_x,

            step_y

        )

        # ----------------------------------------------------
        # 更新標題
        # ----------------------------------------------------

        self.ax_position.set_title(

            f"Lévy Flight Search | Iteration {frame + 1}"

        )

        # ----------------------------------------------------
        # 更新狀態列
        # ----------------------------------------------------

        self.status_label.config(

            text=(

                f"Running | Iteration "

                f"{frame + 1} / "

                f"{self.animation_steps_count}"

            )

        )

        # ----------------------------------------------------
        # 更新畫面
        # ----------------------------------------------------

        self.animation_canvas.draw_idle()

        # ----------------------------------------------------
        # 迭代增加
        # ----------------------------------------------------

        self.current_frame += 1

        # ----------------------------------------------------
        # 排程下一次更新
        # ----------------------------------------------------

        self.animation_job = self.root.after(

            self.animation_delay,

            self.update_animation

        )

    # ========================================================
    # 6. 開始動畫
    # ========================================================

    def start_animation(self):

        """
        主題：
        開始播放動畫。
        """

        if self.animation_running:

            return

        if self.current_frame >= self.animation_steps_count:

            self.current_frame = 0

            self.clear_animation_plot()

        self.animation_running = True

        self.status_label.config(

            text="Running"

        )

        self.update_animation()

    # ========================================================
    # 7. 暫停動畫
    # ========================================================

    def pause_animation(self):

        """
        主題：
        暫停動畫並取消排程。
        """

        self.animation_running = False

        if self.animation_job is not None:

            try:

                self.root.after_cancel(

                    self.animation_job

                )

            except tk.TclError:

                pass

            self.animation_job = None

        self.status_label.config(

            text=f"Paused | Iteration {self.current_frame}"

        )

    # ========================================================
    # 8. 清除動畫圖
    # ========================================================

    def clear_animation_plot(self):

        """
        主題：
        清除動畫軌跡。
        """

        self.position_line.set_data(

            [],

            []

        )

        self.position_point.set_data(

            [],

            []

        )

        self.step_line.set_data(

            [],

            []

        )

        self.ax_position.set_title(

            "Lévy Flight Search Trajectory"

        )

        self.animation_canvas.draw_idle()

    # ========================================================
    # 9. 重設動畫
    # ========================================================

    def reset_animation(self):

        """
        主題：
        將動畫回到初始位置。
        """

        self.pause_animation()

        self.current_frame = 0

        self.clear_animation_plot()

        self.status_label.config(

            text="Reset | Ready"

        )

    # ========================================================
    # 10. 調整動畫速度
    # ========================================================

    def change_speed(self, value):

        """
        主題：
        透過滑桿調整動畫速度。

        數值越小，動畫越快。
        """

        self.animation_delay = max(

            10,

            int(float(value))

        )

        self.status_label.config(

            text=(

                f"Animation Speed: "

                f"{self.animation_delay} ms"

            )

        )

    # ========================================================
    # 11. 儲存分析圖 PNG
    # ========================================================

    def save_analysis_png(self):

        """
        主題：
        儲存四張分析圖。
        """

        file_path = filedialog.asksaveasfilename(

            title="Save Distribution Analysis",

            defaultextension=".png",

            filetypes=[

                ("PNG Image", "*.png")

            ]

        )

        if not file_path:

            return

        try:

            self.analysis_fig.savefig(

                file_path,

                dpi=300,

                bbox_inches="tight"

            )

            messagebox.showinfo(

                "Save Complete",

                "Distribution analysis saved successfully."

            )

        except Exception as error:

            messagebox.showerror(

                "Save Error",

                str(error)

            )

    # ========================================================
    # 12. 儲存動畫 GIF
    # ========================================================

    def save_animation_gif(self):

        """
        主題：
        將 Lévy Flight 動態搜尋儲存為 GIF。

        使用獨立 Figure 產生 GIF，
        不修改目前 Tkinter 動畫的播放狀態。
        """

        file_path = filedialog.asksaveasfilename(

            title="Save Lévy Flight Animation",

            defaultextension=".gif",

            filetypes=[

                ("GIF Animation", "*.gif")

            ]

        )

        if not file_path:

            return

        try:

            # ------------------------------------------------
            # 建立獨立動畫畫布
            # ------------------------------------------------

            gif_fig = Figure(

                figsize=(9, 5),

                dpi=90

            )

            gif_ax = gif_fig.add_subplot(
                111
            )

            # ------------------------------------------------
            # 設定位置軸
            # ------------------------------------------------

            gif_ax.set_xlim(

                0,

                self.animation_steps_count

            )

            position_min = np.min(

                self.animation_positions

            )

            position_max = np.max(

                self.animation_positions

            )

            margin = max(

                1,

                0.1 * (

                    position_max - position_min

                )

            )

            gif_ax.set_ylim(

                position_min - margin,

                position_max + margin

            )

            gif_ax.set_xlabel(
                "Iteration"
            )

            gif_ax.set_ylabel(
                "Search Position"
            )

            gif_ax.grid(
                True,
                alpha=0.3
            )

            # ------------------------------------------------
            # 動畫線條
            # ------------------------------------------------

            gif_line, = gif_ax.plot(

                [],

                [],

                linewidth=1.5,

                label="Search Trajectory"

            )

            gif_point, = gif_ax.plot(

                [],

                [],

                marker="o",

                markersize=7,

                label="Current Position"

            )

            gif_ax.legend()

            # ------------------------------------------------
            # 建立 Agg 畫布
            # ------------------------------------------------

            gif_canvas = FigureCanvasAgg(
                gif_fig
            )

            frames = []

            # ------------------------------------------------
            # 逐格產生 GIF
            # ------------------------------------------------

            for frame in range(

                self.animation_steps_count + 1

            ):

                x_data = self.animation_time[

                    :frame + 1

                ]

                y_data = self.animation_positions[

                    :frame + 1

                ]

                gif_line.set_data(

                    x_data,

                    y_data

                )

                gif_point.set_data(

                    [self.animation_positions[frame]],

                    [self.animation_positions[frame]]

                )

                gif_ax.set_title(

                    f"Lévy Flight Search | Iteration {frame}"

                )

                # 繪製畫面

                gif_canvas.draw()

                # 取得 RGBA 圖片

                image_array = np.asarray(

                    gif_canvas.buffer_rgba()

                )

                # 轉換成 PIL Image

                frame_image = Image.fromarray(

                    image_array

                ).convert("RGB")

                frames.append(
                    frame_image
                )

            # ------------------------------------------------
            # 儲存 GIF
            # ------------------------------------------------

            if frames:

                frames[0].save(

                    file_path,

                    save_all=True,

                    append_images=frames[1:],

                    duration=80,

                    loop=0,

                    optimize=False

                )

            # ------------------------------------------------
            # 關閉 GIF Figure
            # ------------------------------------------------

            plt.close(
                gif_fig
            )

            messagebox.showinfo(

                "Save Complete",

                "GIF animation saved successfully."

            )

        except Exception as error:

            messagebox.showerror(

                "Save Error",

                str(error)

            )

    # ========================================================
    # 13. 關閉程式
    # ========================================================

    def close_application(self):

        """
        主題：
        關閉視窗時停止動畫。
        """

        self.pause_animation()

        self.root.destroy()


# ============================================================
# 第三部分：程式執行入口
# ============================================================

if __name__ == "__main__":

    # 建立 Tkinter 主視窗

    root = tk.Tk()

    # 建立應用程式

    app = LevyFlightApp(
        root
    )

    # 啟動視窗

    root.mainloop()

