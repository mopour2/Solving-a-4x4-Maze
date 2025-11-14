#Mohammad Reza pour Emad
# 11-14-2025
from collections import deque
import json
import math
import os
import cv2, numpy as np, heapq, time, csv, serial.tools.list_ports
from pydobot import Dobot
#--------------------------
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play
import os
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
from openai import OpenAI
import json
import time
import re

# ======= متغیرهای سراسری =======
TextGlobal = None
State = 1
SAMPLE_RATE = 16000
DURATION = 4
start_color = "green"
Flag_Exit = 0
Flag_Voice_Text= 2# 1= Voice  2 =Txet ChatGPT
FlagFileImag=2
Z_Safe= -20
 #============================ Dobot Setup ============================
try:
    from pydobot import Dobot
except Exception:
    Dobot = None  # اجازه می‌دهد بدون ربات هم اجرا و مسیر تولید شود
#-------------------------------------------------------------------------
CAMERA_PORT = 2
DOBOT_PORT = "/dev/ttyACM0"   # ویندوز: "COM3" | لینوکس: "/dev/ttyACM0" یا "/dev/ttyUSB0"
SAVE_FILE = "vision_robot_homography_4aruco.json"

#------------------------------------------------------------------------
H = None
device = None
mask = None
#-------------------------------------------------------------
# ارتفاع صفحه‌ی پلاستیکی (عمق تماس قلم/اندافکتور) — حتماً با ستاپ خودتان Sync کنید
BOARD_Z   = -47                 # ارتفاع تقریبی صفحه (نمونه از کدهای قبلی شما)
PATH_Z    = BOARD_Z + 15        # 15 میلی‌متر بالاتر از صفحه (در بازه 10..20 mm)
TRAVEL_Z  = BOARD_Z + 40        # ارتفاع امن‌تر برای جابجایی بین نقاط
TOOL_R    = 0                   # زاویه R اندافکتور؛ 0 کافی است

HOME      = (220, 0, 150, TOOL_R)  # خانه‌ی امن
SPEED_XY  = 70
SPEED_Z   = 70
# =================== تنظیمات عمومی ===================
WARP_SIZE_X =640 -1
WARP_SIZE_Y =480 -1

KERNEL_SIZE = 1
DILATE_ITER = 1
SMOOTH_WINDOW = 5
SAFE_MARGIN_PX = 10   # فاصله ایمن از دیوار (پیکسل)
REPULSION_STRENGTH = 10
CENTER_WEIGHT = 7.0
Margin_for_delet = 10
# ⚙️ تنظیمات فیزیکی ماز روی میز Dobot
ORIGIN_XY = (300, 5)
CELL_SIZE_MM = 37.5
BOARD_Z = -47
PATH_Z = BOARD_Z + 20



#-------------------------------------------------------------------------------

def find_gate_wall_follow_visual_v5(maze, start, show=True, max_depth=20000):
    """
    منطق:
    1) از start تا نزدیک‌ترین دیوار برو (BFS).
    2) دیوار را در چهار جهت امتداد بده؛ طول و نقطه‌ی انتهایی هر پاره‌خط را نگه دار.
    3) اول در راستای اصلی (طولانی‌تر: افقی/عمودی) تلاش کن: از هر دو انتهای آن، در همان راستا وارد فضای باز شو و تا دیوار بعدی برو؛ اگر فضای باز پیدا شد وسطش = Gate.
    4) اگر در دو جهتِ راستای اصلی فضای باز پیدا نشد:
       - کوتاه‌ترین پاره‌خط را انتخاب کن،
       - در انتهای همان پاره‌خط بایست،
       - گوشه بزن (چرخش ۹۰ درجه) و امتداد جدید دیوار را دنبال کن،
       - سپس دوباره در همان راستا وارد فضای باز شو و Gate را پیدا کن.
    نمایش: پیکسل‌های بررسی = قرمز، Gate = سبز.
    """
    h, w = maze.shape
    vis = cv2.cvtColor((maze * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def is_valid(y, x):
        return 0 <= y < h and 0 <= x < w

    def show_pt(y, x, color=(0,0,255), wait=1):
        if not is_valid(y, x): return
        vis[y, x] = color
        if show:
            cv2.imshow("Gate Finder", vis); cv2.waitKey(wait)

    # ---------- 1) برو تا نزدیک‌ترین دیوار ----------
    from collections import deque
    q = deque([start])
    visited = {start}
    first_wall = None
    while q and len(visited) < max_depth:
        y, x = q.popleft()
        show_pt(y, x, (0,0,255), 1)
        if maze[y, x] == 0:
            first_wall = (y, x); break
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y+dy, x+dx
            if is_valid(ny,nx) and (ny,nx) not in visited:
                visited.add((ny,nx)); q.append((ny,nx))

    if first_wall is None:
        print("❌ دیوار پیدا نشد."); return None

    wy, wx = first_wall

    # ---------- کمک: امتداد دیوار در یک جهت ----------
    def extend_wall(y, x, dy, dx):
        """تا وقتی 0 است حرکت کن؛ نقاط و انتهای پاره‌خط را بده."""
        pts = []
        while is_valid(y, x) and maze[y, x] == 0:
            pts.append((y, x))
            show_pt(y, x, (0,0,255), 1)
            y += dy; x += dx
        end = pts[-1] if pts else (None, None)
        return pts, end  # (فهرست نقاط پاره‌خط، انتها)

    # چهار امتداد از نقطهٔ برخورد
    up_pts,   up_end   = extend_wall(wy, wx, -1, 0)
    down_pts, down_end = extend_wall(wy, wx,  1, 0)
    left_pts, left_end = extend_wall(wy, wx,  0,-1)
    right_pts, right_end=extend_wall(wy, wx,  0, 1)

    # طول‌ها و انتهاها
    segs = {
        "up":    {"len": len(up_pts),    "end": up_end,    "dir":(-1,0)},
        "down":  {"len": len(down_pts),  "end": down_end,  "dir":( 1,0)},
        "left":  {"len": len(left_pts),  "end": left_end,  "dir":( 0,-1)},
        "right": {"len": len(right_pts), "end": right_end, "dir":( 0, 1)},
    }

    vert_total = segs["up"]["len"] + segs["down"]["len"]
    hori_total = segs["left"]["len"] + segs["right"]["len"]
    main_axis  = "vertical" if vert_total >= hori_total else "horizontal"

    # ---------- کمک: حرکت در فضای بازِ «پس از انتهای دیوار» تا دیوار بعدی ----------
    def walk_free_from(end_pt, dy, dx):
        """از انتهای پاره‌خط، یک پیکسل جلوتر وارد فضای باز شو و تا برخورد دیوار برو؛ لیست فضای باز را بده."""
        if end_pt[0] is None: return []
        y, x = end_pt[0]+dy, end_pt[1]+dx
        free = []
        while is_valid(y, x) and maze[y, x] == 1:
            free.append((y, x))
            show_pt(y, x, (0,0,255), 1)
            y += dy; x += dx
        return free

    # ---------- 3) اول در راستای اصلی تلاش کن (دو جهتش) ----------
    def try_main_axis():
        if main_axis == "vertical":
            # بالا/پایین → سپس همان راستا به فضای باز
            free1 = walk_free_from(segs["up"]["end"],   -1, 0)
            if free1:
                return free1
            free2 = walk_free_from(segs["down"]["end"],  1, 0)
            return free2
        else:
            free1 = walk_free_from(segs["left"]["end"],  0, -1)
            if free1:
                return free1
            free2 = walk_free_from(segs["right"]["end"], 0,  1)
            return free2

    free_line = try_main_axis()

    # ---------- 4) اگر در راستای اصلی هیچ فضای بازی پیدا نشد:
    #     «کوتاه‌ترین پاره‌خط» را انتخاب کن، در انتهایش بایست، گوشه بزن (۹۰°) و دوباره تلاش کن.
    if not free_line:
        # کوتاه‌ترین پاره‌خط واقعی (len>0)
        nonzero = [(k,v) for k,v in segs.items() if v["len"]>0]
        if not nonzero:
            print("⚠️ هیچ پاره‌خط معناداری از دیوار نداریم."); return None
        shortest_key, shortest = min(nonzero, key=lambda kv: kv[1]["len"])
        end_y, end_x = shortest["end"]; sdy, sdx = shortest["dir"]

        # گوشه زدن: چرخش ۹۰ درجه روی انتهای پاره‌خط کوتاه‌تر
        # اگر کوتاه‌ترین عمودی بود → افقی‌ها را امتحان کن، و بالعکس.
        if shortest_key in ["up","down"]:
            # عمودی بود → افقی‌ها
            cand_dirs = [(0,-1),(0,1)]
        else:
            # افقی بود → عمودی‌ها
            cand_dirs = [(-1,0),(1,0)]

        # اول امتداد دیوار جدید در جهت‌های عمود را پیدا کن، بعد از انتهای آن وارد فضای باز شو
        found = []
        for pdy, pdx in cand_dirs:
            # امتداد دیوار در جهت عمود (کنار گوشه)
            wall2_pts, wall2_end = extend_wall(end_y + pdy, end_x + pdx, pdy, pdx)
            if wall2_pts:
                # از انتهای دیوار جدید، در همان راستا وارد فضای باز شو
                free_try = walk_free_from(wall2_end, pdy, pdx)
                if free_try:
                    found = free_try; break
                # اگر بلافاصله فضای باز نبود، برعکس همین راستا را هم امتحان کن
                free_try2 = walk_free_from(wall2_end, -pdy, -pdx)
                if free_try2:
                    found = free_try2; break

        free_line = found

    # ---------- خروجی Gate ----------
    if not free_line:
        print("⚠️ فضای آزاد بین دو دیوار پیدا نشد."); return None

    mid = len(free_line)//2
    gy, gx = free_line[mid]
    show_pt(gy, gx, (0,255,0), 0)
    if show:
        cv2.waitKey(0); cv2.destroyAllWindows()
    print(f"✅ Gate @ ({gy}, {gx})")
    return (gy, gx)

#-----------------------------------------------------------------
def astar_safe_visual(maze, start, end, show=True):
    """
    نسخه‌ی تصویری از A*:
    هر پیکسل که بررسی می‌شود به رنگ قرمز در تصویر نمایش داده می‌شود.
    """
    h, w = maze.shape
    moves = [(-1,0),(1,0),(0,-1),(0,1)]

    # محاسبه‌ی نقشه فاصله از دیوار
    dist = cv2.distanceTransform((maze*255).astype(np.uint8), cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (9,9), 0)
    dist_norm = cv2.normalize(dist, None, 0.0, 1.0, cv2.NORM_MINMAX)
    wall_cost = np.exp(-CENTER_WEIGHT * dist_norm)

    # برای نمایش رنگی (RGB)
    vis = cv2.cvtColor((maze * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    open_set = [(0, start)]
    came_from = {}
    g = {start: 0}
    f = {start: np.linalg.norm(np.array(start) - np.array(end))}

    while open_set:
        _, current = heapq.heappop(open_set)

        # اگر به هدف رسیدیم → مسیر بساز و نمایش بده
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()

            # مسیر نهایی آبی
            for (y, x) in path:
                vis[y, x] = (255, 0, 0)
                if show:
                    cv2.imshow("A* Path Progress", vis)
                    cv2.waitKey(15)
            if show:
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return path, dist

        # همسایه‌ها
        for dx, dy in moves:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx < h and 0 <= ny < w and maze[nx, ny] == 1:

                penalty = 1.0 + 6 * wall_cost[nx, ny]
                if dist[nx, ny] < SAFE_MARGIN_PX:
                    penalty += (SAFE_MARGIN_PX - dist[nx, ny]) * 2

                new_g = g[current] + penalty
                if (nx, ny) not in g or new_g < g[(nx, ny)]:
                    g[(nx, ny)] = new_g
                    f_val = new_g + np.linalg.norm(np.array([nx, ny]) - np.array(end))
                    heapq.heappush(open_set, (f_val, (nx, ny)))
                    came_from[(nx, ny)] = current

                    # 🎨 نمایش پیکسل در حال بررسی (قرمز)
                    vis[nx, ny] = (0, 0, 255)
                    if show:
                        cv2.imshow("A* Path Progress", vis)
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:
                         cv2.destroyAllWindows()
                         return None, dist
    if show:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return None, dist

#=================== تابع A* اصلاح‌شده ===================
def astar_safe_visual2(maze, start, end, show=True):
    """
    نسخه پیشرفته A* با Distance Transform و دافعه از دیوارها.
    """
 

    h, w = maze.shape
    moves = [(-1,0),(1,0),(0,-1),(0,1)]
    dist = cv2.distanceTransform((maze*255).astype(np.uint8), cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (9,9), 0)
    dist_norm = cv2.normalize(dist, None, 0.0, 1.0, cv2.NORM_MINMAX)

# برای نمایش رنگی (RGB)
    vis = cv2.cvtColor((maze * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    wall_cost = np.exp(-CENTER_WEIGHT * dist_norm)  # نزدیکی به دیوار = هزینه زیاد
    open_set = [(0, start)]
    came_from = {}
    g = {start: 0}
    f = {start: np.linalg.norm(np.array(start) - np.array(end))}
    edge_penalty = 0.0
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            
             # مسیر نهایی آبی
            for (y, x) in path:
                vis[y, x] = (255, 0, 0)
                if show:
                    cv2.imshow("A* Path Progress", vis)
                    cv2.waitKey(15)
            if show:
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return path, dist
        
        for dx, dy in moves:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx < h and 0 <= ny < w and maze[nx, ny] == 1:
                
                if dist[nx, ny] == 0:
                  edge_penalty = 0.0
                  if (nx < BORDER_BAN_WIDTH or ny < BORDER_BAN_WIDTH or
                         nx >= h - BORDER_BAN_WIDTH or ny >= w - BORDER_BAN_WIDTH):
                      edge_penalty = 1000.0  # جریمهٔ خیلی بزرگ
                penalty = 1.0 + 6 * wall_cost[nx, ny] + edge_penalty
                # فاصله از دیوار — اگر کمتر از حد ایمن است → هزینه زیاد
                #penalty = 1.0 + 6 * wall_cost[nx, ny]
                if dist[nx, ny] < SAFE_MARGIN_PX:
                    penalty += (SAFE_MARGIN_PX - dist[nx, ny]) * 2

                # جلوگیری از بریدن گوشه
                if current in came_from:
                    px, py = came_from[current]
                    if (dx, dy) != (current[0]-px, current[1]-py):
                        penalty += 1.0

                new_g = g[current] + penalty
                if (nx, ny) not in g or new_g < g[(nx, ny)]:
                    g[(nx, ny)] = new_g
                    f_val = new_g + np.linalg.norm(np.array([nx,ny]) - np.array(end))
                    heapq.heappush(open_set, (f_val, (nx, ny)))
                    came_from[(nx, ny)] = current
                 
                  # 🎨 نمایش پیکسل در حال بررسی (قرمز)
                    vis[nx, ny] = (0, 0, 255)
                    if show:
                        cv2.imshow("A* Path Progress", vis)
                        key = cv2.waitKey(2) & 0xFF
                        if key == 27:
                         cv2.destroyAllWindows()
                         return None, dist
    if show:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return None, dist
 #=================== تابع A* اصلاح‌شده ===================
def astar_safe2(maze, start, end):
    """
    نسخه پیشرفته A* با Distance Transform و دافعه از دیوارها.
    """
    h, w = maze.shape
    moves = [(-1,0),(1,0),(0,-1),(0,1)]
    dist = cv2.distanceTransform((maze*255).astype(np.uint8), cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (9,9), 0)
    dist_norm = cv2.normalize(dist, None, 0.0, 1.0, cv2.NORM_MINMAX)

    wall_cost = np.exp(-CENTER_WEIGHT * dist_norm)  # نزدیکی به دیوار = هزینه زیاد
    open_set = [(0, start)]
    came_from = {}
    g = {start: 0}
    f = {start: np.linalg.norm(np.array(start) - np.array(end))}
    edge_penalty = 0.0
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, dist

        for dx, dy in moves:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx < h and 0 <= ny < w and maze[nx, ny] == 1:
                
                if dist[nx, ny] == 0:
                  edge_penalty = 0.0
                  if (nx < BORDER_BAN_WIDTH or ny < BORDER_BAN_WIDTH or
                         nx >= h - BORDER_BAN_WIDTH or ny >= w - BORDER_BAN_WIDTH):
                      edge_penalty = 1000.0  # جریمهٔ خیلی بزرگ
                penalty = 1.0 + 6 * wall_cost[nx, ny] + edge_penalty
                # فاصله از دیوار — اگر کمتر از حد ایمن است → هزینه زیاد
                #penalty = 1.0 + 6 * wall_cost[nx, ny]
                if dist[nx, ny] < SAFE_MARGIN_PX:
                    penalty += (SAFE_MARGIN_PX - dist[nx, ny]) * 2

                # جلوگیری از بریدن گوشه
                if current in came_from:
                    px, py = came_from[current]
                    if (dx, dy) != (current[0]-px, current[1]-py):
                        penalty += 1.0

                new_g = g[current] + penalty
                if (nx, ny) not in g or new_g < g[(nx, ny)]:
                    g[(nx, ny)] = new_g
                    f_val = new_g + np.linalg.norm(np.array([nx,ny]) - np.array(end))
                    heapq.heappush(open_set, (f_val, (nx, ny)))
                    came_from[(nx, ny)] = current
    return None, dist

# =================== تابع A* اصلاح‌شده ===================
def astar_safe(maze, start, end):
    """
    نسخه پیشرفته A* با Distance Transform و دافعه از دیوارها.
    """
    h, w = maze.shape
    moves = [(-1,0),(1,0),(0,-1),(0,1)]
    dist = cv2.distanceTransform((maze*255).astype(np.uint8), cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (9,9), 0)
    dist_norm = cv2.normalize(dist, None, 0.0, 1.0, cv2.NORM_MINMAX)

    wall_cost = np.exp(-CENTER_WEIGHT * dist_norm)  # نزدیکی به دیوار = هزینه زیاد
    open_set = [(0, start)]
    came_from = {}
    g = {start: 0}
    f = {start: np.linalg.norm(np.array(start) - np.array(end))}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, dist

        for dx, dy in moves:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx < h and 0 <= ny < w and maze[nx, ny] == 1:
                
                if dist[nx, ny] == 0:
                  continue

                # فاصله از دیوار — اگر کمتر از حد ایمن است → هزینه زیاد
                penalty = 1.0 + 6 * wall_cost[nx, ny]
                if dist[nx, ny] < SAFE_MARGIN_PX:
                    penalty += (SAFE_MARGIN_PX - dist[nx, ny]) * 2

                # جلوگیری از بریدن گوشه
                if current in came_from:
                    px, py = came_from[current]
                    if (dx, dy) != (current[0]-px, current[1]-py):
                        penalty += 1.0

                new_g = g[current] + penalty
                if (nx, ny) not in g or new_g < g[(nx, ny)]:
                    g[(nx, ny)] = new_g
                    f_val = new_g + np.linalg.norm(np.array([nx,ny]) - np.array(end))
                    heapq.heappush(open_set, (f_val, (nx, ny)))
                    came_from[(nx, ny)] = current
    return None, dist

# =================== Smooth Path Function ===================
def smooth_path(path, window=5):
    if len(path) < window:
        return path
    smoothed = []
    for i in range(len(path)):
        y_vals = [path[j][0] for j in range(max(0,i-window), min(len(path),i+window))]
        x_vals = [path[j][1] for j in range(max(0,i-window), min(len(path),i+window))]
        smoothed.append((int(np.mean(y_vals)), int(np.mean(x_vals))))
    return smoothed
#======================================================================
def compress_straight_segments(path_px):
    """
    ورودی: path_px لیستی از نقاط پیکسلی به فرم [(y,x), (y,x), ...]
    خروجی: فقط نقاطِ تغییر جهت + اولین و آخرین نقطه (نقاط وسطِ خطوط مستقیم حذف می‌شوند)
    """
    if not path_px or len(path_px) < 3:
        return path_px[:]

    keep = [path_px[0]]
    # جهت گام اول را به بردار با مؤلفه‌های -1/0/1 نگاشت می‌کنیم
    dy_prev = path_px[1][0] - path_px[0][0]
    dx_prev = path_px[1][1] - path_px[0][1]
    dir_prev = (np.sign(dy_prev), np.sign(dx_prev))

    for i in range(1, len(path_px) - 1):
        dy = path_px[i+1][0] - path_px[i][0]
        dx = path_px[i+1][1] - path_px[i][1]
        dir_now = (np.sign(dy), np.sign(dx))

        # اگر جهت عوض شد، این نقطه یک «نقطه‌ی شکست/چرخش» است و باید حفظ شود
        if dir_now != dir_prev:
            keep.append(path_px[i])
            dir_prev = dir_now

    keep.append(path_px[-1])
    return keep

# =================== Utility Functions ===================


def find_token_center(img, color):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if color == "green":
        lower, upper = np.array([35,50,50]), np.array([85,255,255])
        mask = cv2.inRange(hsv, lower, upper)
    else:
        lower1, upper1 = np.array([0, 70, 40]),  np.array([15, 255, 255])
        lower2, upper2 = np.array([160, 70, 40]), np.array([180, 255, 255])

        mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                              cv2.inRange(hsv, lower2, upper2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5,5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        area = cv2.contourArea(c)
        if M["m00"] != 0 and area > 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            r = int(max(8, np.sqrt(area/np.pi)))
            return (cx, cy), mask, r
    return None, mask, None


import cv2
import numpy as np

def find_token_center_and_draw(img):
    frame = img.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ====== Red Detection ======
    lower1, upper1 = np.array([0, 50, 40]), np.array([15, 255, 255])
    lower2, upper2 = np.array([160, 50, 40]), np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    red_center = None
    cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        (rx, ry), rr = cv2.minEnclosingCircle(c)
        if rr > 3:
            red_center = (int(rx), int(ry))
            cv2.circle(frame, red_center, int(rr), (0, 0, 255), 2)   # 🔴 دایره قرمز دورش بکش
            cv2.circle(frame, red_center, 3, (0, 0, 255), -1)       # نقطه‌ی مرکز

    # ====== Green Detection ======
    v_mean = np.mean(hsv[:, :, 2])
    if v_mean > 180:
        s_min, v_min = 10, 40
        h_low, h_high = 25, 95
    elif v_mean < 80:
        s_min, v_min = 40, 30
        h_low, h_high = 35, 90
    else:
        s_min, v_min = 25, 35
        h_low, h_high = 30, 90

    green_mask = cv2.inRange(hsv, np.array([h_low, s_min, v_min]), np.array([h_high, 255, 255]))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    green_center = None
    cnts, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        possible_centers = []
        for c in cnts:
            (gx, gy), gr = cv2.minEnclosingCircle(c)
            if gr > 3:
                possible_centers.append((int(gx), int(gy)))

        if red_center is not None and possible_centers:
            # دورترین دایره نسبت به قرمز را سبز در نظر بگیر
            dists = [np.linalg.norm(np.array(red_center) - np.array(pt)) for pt in possible_centers]
            green_center = possible_centers[np.argmax(dists)]
            cv2.circle(frame, green_center, int(gr), (0, 255, 0), 2)   # 🟢 دایره سبز دورش بکش
            cv2.circle(frame, green_center, 3, (0, 255, 0), -1)       # نقطه‌ی مرکز

    return frame, red_center, green_center


def find_token_center11(img):
    frame = img.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ====== Red Detection ======
    lower1, upper1 = np.array([0, 50, 40]), np.array([15, 255, 255])
    lower2, upper2 = np.array([160, 50, 40]), np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    red_center = None
    cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        (rx, ry), rr = cv2.minEnclosingCircle(c)
        if rr > 3:
            red_center = (int(rx), int(ry))

    # ====== Green Detection ======
    v_mean = np.mean(hsv[:, :, 2])
    if v_mean > 180:
        s_min, v_min = 10, 40
        h_low, h_high = 25, 95
    elif v_mean < 80:
        s_min, v_min = 40, 30
        h_low, h_high = 35, 90
    else:
        s_min, v_min = 25, 35
        h_low, h_high = 30, 90

    green_mask = cv2.inRange(hsv, np.array([h_low, s_min, v_min]), np.array([h_high, 255, 255]))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    green_center = None
    cnts, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        # مرکز سبز را جدا از قرمز انتخاب کن (در فاصله معقول)
        possible_centers = []
        for c in cnts:
            (gx, gy), gr = cv2.minEnclosingCircle(c)
            if gr > 3:
                possible_centers.append((int(gx), int(gy)))

        if red_center is not None and possible_centers:
            # نزدیک‌ترین یا دورترین نقطه نسبت به قرمز (معمولاً یکی سبز است)
            dists = [np.linalg.norm(np.array(red_center) - np.array(pt)) for pt in possible_centers]
            green_center = possible_centers[np.argmax(dists)]  # دورترین دایره را سبز فرض می‌کنیم

    return red_center, green_center




def find_token_center1(img, color):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  if color == "green":
    v_mean = np.mean(hsv[:, :, 2])  # میانگین روشنایی تصویر

    if v_mean > 180:   # نور زیاد → سبز روشن یا فسفری
        s_min, v_min = 10, 40
        h_low, h_high = 25, 95
    elif v_mean < 80:  # نور کم → سبز تیره
        s_min, v_min = 40, 30
        h_low, h_high = 35, 90
    else:              # نور معمولی
        s_min, v_min = 25, 35
        h_low, h_high = 30, 90

    lower = np.array([h_low, s_min, v_min])
    upper = np.array([h_high, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)


  else:  # 🔴 Red
     # دو محدوده برای قرمز (قرمز تیره و قرمز روشن)
      lower1 = np.array([0, 50, 40])
      upper1 = np.array([15, 255, 255])
      lower2 = np.array([160, 50, 40])
      upper2 = np.array([180, 255, 255])
    # ✅ فقط یک بار ماسک‌ها رو ترکیب کن
      mask1 = cv2.inRange(hsv, lower1, upper1)
      mask2 = cv2.inRange(hsv, lower2, upper2)
      mask = cv2.bitwise_or(mask1, mask2)

# ==============================
# 🧹 تمیز کردن ماسک و حذف نویز
# ==============================
  kernel = np.ones((5, 5), np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # حذف نویزهای کوچک 
  mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel) # پر کردن نواحی ناقص

# ==============================
# 🟢 پیدا کردن کانتور رنگ
# ==============================
  cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  if cnts:
        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        area = cv2.contourArea(c)
        if M["m00"] != 0 and area > 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            r = int(max(8, np.sqrt(area/np.pi)))
            return (cx, cy), mask, r
  return None, mask, None

#======================================================================
def remove_tokens_from_binary(binary, centers_radii, margin=6):
    cleaned = binary.copy()
    for (cx, cy, r) in centers_radii:
        if cx and cy and r:
            R = int(r + margin)
            cv2.circle(cleaned, (cx, cy), R, (0, 0, 0), -1)
    return cleaned
#======================================================================
def pixel_to_dobot(x_px, y_px):
    # ⚙️ ابعاد واقعی ماز روی میز (میلی‌متر)
    MAZE_MM_X = 200
    MAZE_MM_Y = 220

    # 🧭 مرکز ماز روی میز (میلی‌متر)
   # ORIGIN_XY = (290, 0)

    # 📏 نسبت تبدیل پیکسل به میلی‌متر
    SCALE_X = MAZE_MM_X / WARP_SIZE_X
    SCALE_Y = MAZE_MM_Y / WARP_SIZE_Y

    # 🔄 تبدیل مختصات پیکسل به مختصات ربات
    X = ORIGIN_XY[0] + (y_px - WARP_SIZE_X/2) * SCALE_X
    Y = ORIGIN_XY[1] + (x_px - WARP_SIZE_Y/2) * SCALE_Y
    print("X=",X," ","y=",Y," ","x=",y_px,"y=",x_px)
    return X, Y, PATH_Z


# =================== Calibration ===================
_calib_pts = []
def _on_mouse(event, x, y, flags, param):
    global _calib_pts
    if event == cv2.EVENT_LBUTTONDOWN and len(_calib_pts) < 4:
        _calib_pts.append((x, y))
        print(f"[Calib] Point {len(_calib_pts)}: ({x}, {y})")
#======================================================================
def calibrate_board(cap):
    global _calib_pts
    _calib_pts = []
    win = "Board Calib"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _on_mouse)
    print("[Calib] Click 4 corners TL, TR, BR, BL then press Enter.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        disp = frame.copy()
        for i, p in enumerate(_calib_pts):
            cv2.circle(disp, p, 6, (0, 255, 255), -1)
            cv2.putText(disp, str(i + 1), (p[0] + 5, p[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            _calib_pts = []
        elif key == ord('q'):
            cv2.waitKey(100)
  
            cv2.destroyWindow(win)
            return None
        elif key in (13, 10) and len(_calib_pts) == 4:
            #src = np.array(_calib_pts, dtype=np.float32)
            
          # src = np.array([[0, 0], [WARP_SIZE_X, 0],
          #                  [WARP_SIZE_X, WARP_SIZE_Y], [0, WARP_SIZE_Y]], dtype=np.float32)
           
            src = np.array([[WARP_SIZE_X, WARP_SIZE_Y], [0, WARP_SIZE_Y],
                            [0, 0], [WARP_SIZE_X, 0]], dtype=np.float32)    
            dst = np.array([[0, 0], [WARP_SIZE_X, 0],
                            [WARP_SIZE_X, WARP_SIZE_Y], [0, WARP_SIZE_Y]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            np.save("calibration_matrix.npy", M)
            print("[Calib] Saved calibration to calibration_matrix.npy")
            cv2.destroyAllWindows()
            cv2.waitKey(100)
   

            return M
#======================================================================
def load_calibration():
    try:
        M = np.load("calibration_matrix.npy")
        print("[Calib] Loaded existing calibration.")
        return M
    except Exception:
        print("[Calib] No saved file found.")
        return None
#======================================================================
def calibrate_Papar(cap):
    global _calib_pts
    _calib_pts = []
    win = "Board Calib"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _on_mouse)
    print("[Calib] Click 4 corners TL, TR, BR, BL then press Enter.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        disp = frame.copy()
        
        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
           break
             
    cv2.destroyWindow(win)
    return key
 

# ============ ArUco Detection ============
def detect_aruco_markers(cap, needed_ids=(0,1,2,3)):
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        use_new = True
    except:
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters_create()
        use_new = False

    centers = {}
    print("🎯 Detecting 4 ArUco markers (IDs 0,1,2,3)... Press ESC to continue.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # 🌀 Flip 180° before processing
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if use_new:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            ids = ids.flatten()
            for i, marker_id in enumerate(ids):
                if marker_id in needed_ids:
                    c = corners[i][0]
                    center = c.mean(axis=0)
                    centers[marker_id] = center
                    cv2.polylines(frame, [c.astype(int)], True, (0,255,0), 2)
                    cv2.putText(frame, f"ID {marker_id}",
                                tuple(c[0].astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("ArUco Detection", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyWindow("ArUco Detection")
    if len(centers) < 4:
        print(f"⚠️ Only {len(centers)} markers detected. Make sure all 4 are visible.")
    else:
        print("✅ All 4 markers detected.")
    return centers

#======================================================================
 # --- Convert pixel to robot coordinates ---
def image_to_robot(x_img, y_img):
        global H
        p = np.array([[[x_img, y_img]]], dtype=float)
        p_r = cv2.perspectiveTransform(p, H)[0][0]
        return p_r[0], p_r[1],PATH_Z
#======================================================================
def getkey():
    while True:
        k = cv2.waitKey(1) & 0xFF
        if ord('a') <= k <= ord('z') or k in [13, 27]:
            break
# ============================ Robot Driver ===========================
def Get_calibrate_H(cap):
     # --- Calibration / Load existing homography ---
    global H, device, mask   # 👈 اعلام اینکه از نسخه‌ی global استفاده می‌کنیم
    
    # --- Calibration / Load existing homography ---
    if os.path.exists(SAVE_FILE):
        print(f"📁 Calibration file found: {SAVE_FILE}")
        with open(SAVE_FILE, "r") as f:
            data = json.load(f)
        H = np.array(data["homography"], dtype=float)
        print("✅ Calibration file loaded successfully.")
    else:
        centers = detect_aruco_markers(cap)
        if len(centers) < 4:
            print("⚠️ Not all 4 markers detected. Exiting.")
            return

        aruco_real = {}
        for marker_id in [0,1,2,3]:
            input(f"\n👉 Move the robot tool tip to the center of ArUco ID={marker_id} and press Enter...")
            pose, _ = device.get_pose()
            x, y, z, r = pose
            aruco_real[marker_id] = np.array([x, y], dtype=float)
            print(f"📍 Real coordinates for marker {marker_id}: {aruco_real[marker_id].tolist()}")

        device.move_to(*HOME)

        img_pts = np.array([centers[i] for i in sorted(centers.keys())], dtype=float)
        real_pts = np.array([aruco_real[i] for i in sorted(aruco_real.keys())], dtype=float)
        H, mask = cv2.findHomography(img_pts, real_pts, cv2.RANSAC, 2.0)
        print(f"✅ Homography computed ({int(mask.sum())}/{len(mask)} inliers).")

        data = {
            "homography": H.tolist(),
            "aruco_img_pts": img_pts.tolist(),
            "aruco_real_pts": real_pts.tolist()
        }
        with open(SAVE_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"💾 Calibration data saved to {SAVE_FILE}.")
#-------------------------------------------------------------------------
def Clean_Nois(binary):
    # حذف نویزها و خطوط نازک
    kernel_thick = np.ones((3, 3), np.uint8)  # هسته‌ی کوچک‌تر برای ظرافت بیشتر

# ابتدا نویز و نقاط کوچک حذف می‌شن
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_thick, iterations=1)

# بعد با erode خطوط نازک‌تر از 2 پیکسل حذف می‌شن
    binary = cv2.erode(binary, kernel_thick, iterations=1)

# حالا شکاف‌های جزئی پر می‌شن تا ساختار اصلی ماز سالم بمونه
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_thick, iterations=2)
    return binary
#--------------------------------------------------------------------
def Clean_Nois2(binary):
# حذف نویز بدون نازک شدن دیوارها
 num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
 sizes = stats[1:, -1]
 min_size = 150
 binary_clean = np.zeros_like(binary)
 for i in range(0, num_labels - 1):
     if sizes[i] >= min_size:
         binary_clean[labels == i + 1] = 255
 binary = binary_clean
 return binary
#-----------------------------------------------------
def Delet_space(binary, gates, gate_radius=300):
   
    img = (binary > 127).astype(np.uint8) * 255
    h, w = img.shape
    visited = np.zeros((h, w), np.uint8)

    # ساخت ماسک محافظ برای Gateها
    protect_mask = np.zeros_like(img)
    for (gy, gx) in gates:
        cv2.circle(protect_mask, (int(gx), int(gy)), gate_radius, 255, -1)

    q = deque()

    # از پیکسل‌های سیاه در مرز شروع کن
    for x in range(w):
        if img[0, x] == 0: q.append((0, x))
        if img[h-1, x] == 0: q.append((h-1, x))
    for y in range(h):
        if img[y, 0] == 0: q.append((y, 0))
        if img[y, w-1] == 0: q.append((y, w-1))

    # BFS برای پر کردن بیرون
    while q:
        y, x = q.popleft()
        if y < 0 or y >= h or x < 0 or x >= w:
            continue
        if visited[y, x]:
            continue
        if img[y, x] == 255:
            continue
        if protect_mask[y, x] == 255:   # اگر نزدیک Gate بود، پر نکن
            continue

        visited[y, x] = 1
        img[y, x] = 255  # بیرون → سفید

        q.append((y-1, x))
        q.append((y+1, x))
        q.append((y, x-1))
        q.append((y, x+1))

    return img
import cv2
import numpy as np

def remove_circles(img, red_center, green_center, radius=20):
    """
    حذف دو دایره (قرمز و سبز) از تصویر بدون تغییر فایل اصلی
    
    پارامترها:
        img: تصویر اصلی (BGR)
        red_center: مختصات مرکز دایره قرمز (x, y)
        green_center: مختصات مرکز دایره سبز (x, y)
        radius: شعاع ناحیه‌ای که باید پاک شود (پیش‌فرض 20 پیکسل)
    """
    frame = img.copy()
    h, w= frame.shape

    # رنگ پس‌زمینه را از میانگین پیکسل‌های اطراف بگیر
    bg_color = np.mean(frame, axis=(0,1)).astype(np.uint8)

    # اگر مرکز قرمز مشخص است
    if red_center is not None:
        cv2.circle(frame, red_center, radius, bg_color.tolist(), -1)

    # اگر مرکز سبز مشخص است
    if green_center is not None:
        cv2.circle(frame, green_center, radius, bg_color.tolist(), -1)

    return frame


#-----------------------------------------------------------------------

def get_frame_from_file(path):
    """
    یک تصویر از فایل می‌خواند و آن را به صورت یک فریم بازمی‌گرداند.
    برمی‌گرداند: (ret, frame) تا شبیه به cv2.VideoCapture.read رفتار کند.
    """
    if not os.path.exists(path):
        return False, None

    frame = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    if frame is None:
        return False, None
    return True, frame

#-----------------------------------------------------------------------------
def radial_roundness(cnt, center):
    """ نسبت std/mean فاصلهٔ نقاط کانتور تا مرکز دایره """
    cx, cy = center
    pts = cnt.reshape(-1, 2)
    d = np.sqrt((pts[:,0]-cx)**2 + (pts[:,1]-cy)**2)
    if d.size < 10: 
        return 1.0
    return float(np.std(d) / (np.mean(d) + 1e-6))

def detect_circles_shape_first(img, min_r=14, max_r=60):
    """
    فقط دایره‌های واقعی را برمی‌گرداند؛
    خروجی: [(x, y, r, color)]
    """
    vis = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # لبه‌ها + بستن شکاف‌ها
    edges = cv2.Canny(blur, 60, 140)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), 1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)

    # کانتورها با سلسله‌مراتب
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    results = []
    for i, cnt in enumerate(contours):
        # فقط کانتورهای بدون بچه (hierarchy[0][i][2] == -1)
        if hierarchy[0][i][2] != -1:
            continue

        area = cv2.contourArea(cnt)
        if area < 120:
            continue

        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        r = float(r)
        if r < min_r or r > max_r:
            continue

        # چندضلعی‌بودن؛ مربع‌ها معمولاً ~4 رأس دارند
        eps = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) <= 3:      # مثلث/شکل تیز
            continue
        if len(approx) == 4:      # مربع/مستطیل → رد
            continue

        # گردی (circularity)
        per = cv2.arcLength(cnt, True)
        circularity = 4 * math.pi * (area / (per*per + 1e-6))
        if circularity < 0.78:
            continue

        # سنجهٔ دایره‌بودن شعاعی (خیلی کلیدی برای حذف مربع)
        rr = radial_roundness(cnt, (cx, cy))
        if rr > 0.12:  # هرچه کوچک‌تر → دایره‌تر (0.05~0.12 مناسب)
            continue

        # --- فقط حالا رنگ را برچسب می‌زنیم ---
        mask = np.zeros_like(gray)
        cv2.circle(mask, (int(cx), int(cy)), int(r), 255, -1)
        b, g, rv = cv2.mean(img, mask=mask)[:3]
        v = (b+g+rv)/3.0
        if rv > g + 35 and rv > b + 35:
            color = "red";   col = (0,0,255)
        elif g > rv + 35 and g > b + 35:
            color = "green"; col = (0,255,0)
        elif v < 65:
            color = "black"; col = (0,0,0)
        else:
            color = "unknown"; col = (255,255,0)

        # رسم خروجی
        cv2.circle(vis, (int(cx), int(cy)), int(r), col, 3)
        cv2.circle(vis, (int(cx), int(cy)), 2, (255,255,255), -1)
        cv2.putText(vis, color, (int(cx)-22, int(cy)-int(r)-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        results.append((int(cx), int(cy), int(r), color))

    return results, vis

# =================== Solve Maze ===================
def solve_maze_and_get_path(run_robot):
    global H, device, mask,start_color 

    #start_color = "green"
    end_color = "red" if start_color == "green" else "green"
#---------------------------------------------
    if run_robot and Dobot is not None:
        device = Dobot(port=DOBOT_PORT)
        device.speed(SPEED_XY, SPEED_Z)
        device.move_to(*HOME)
#---------------------------------------------
    
    cap = cv2.VideoCapture(CAMERA_PORT)
    time.sleep(2)
    Get_calibrate_H(cap)
#---------------------------------------------
    M = load_calibration()
    if M is None:
        M = calibrate_board(cap)
        if M is None:
            print("❌ Calibration failed.")
            return
#---------------------------------------------

    calibrate_Papar(cap)
    ret, img = cap.read()
    if not ret:
        print("❌ Camera not available.")
        return
#---------------------------------------------
    if FlagFileImag==1:
      ret, img = get_frame_from_file("/home/mohammadreza/Desktop/Mazi/AIVoice/2.jpg")
      if not ret:
        print("خطا: فایل باز نشد.")  
      else:
         cv2.imshow("frame", img)
         cv2.waitKey(0)
         cv2.destroyAllWindows()
    
#---------------------------------------------
    # 🔄 Warp با فیلتر نرم برای جلوگیری از نویز لبه
    warped = cv2.warpPerspective(img, M, (WARP_SIZE_X, WARP_SIZE_Y))
    warped = cv2.GaussianBlur(warped, (3, 3), 0)
    cv2.imshow("Warped Maze", warped)
    getkey()
#---------------------------------------------
    
    circles, out = detect_circles_shape_first(warped)
    print("✅ دایره‌های نهایی:")
    for c in circles:
        print(c)

    #cv2.circle(vis, (int(cx), int(cy)), int(r), col, 3)     
   # cv2.imshow("Circles (shape-only, anti-square)", out)
  #  cv2.waitKey(0)
   # cv2.destroyAllWindows()

    if len(circles)>1 :
     for (x, y, r, color) in circles:
      if color == "red":
         red_center = (x, y)
         r_Red=r
      else :
         green_center = (x, y)
         r_Green=r
    else :
        start, _, r_start = find_token_center(warped, start_color)
        print(start)
        end, _, r_end = find_token_center(warped, end_color)
        print(end)
        if not start or not end:
           print("❌ Could not detect both Stat And ENd.")
           return
    if len(circles)>1 :
     if start_color=="red" :
       start= red_center 
       r_start=r_Red
       end =green_center 
       r_end =r_Green
     else :
       start= green_center 
       r_start=r_Green
       end =red_center 
       r_end = r_Red
    
    cv2.circle(warped, (int(start[0]), int(start[1])), int(r_start), (255,255,255), 3)     
    cv2.imshow("Circles test", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#---------------------------------------------
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 4)

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.dilate(binary, kernel, iterations=DILATE_ITER)
  #---------------------------------------------

  #---------------------------------------------
    # 🧹 حذف توکن‌ها

    cv2.circle(binary, (int(start[0]), int(start[1])), int(r_start), (255,255,255), 3)     
    cv2.imshow("Circles test", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cleaned = remove_tokens_from_binary(binary, [(start[0], start[1], r_start),
                                            (end[0], end[1], r_end)], margin=Margin_for_delet)
    #cleaned= remove_circles(binary, start, end, radius=25)
    #cleaned= binary
   #--------------------------------------------- 
    cleaned=Clean_Nois2(cleaned)
    cleaned=Clean_Nois2(cleaned)
    cleaned=Clean_Nois2(cleaned)
   

#---------------------------------------------
    cv2.imshow("Binary Maze Cleaned", cleaned)
    getkey()
    sy, sx, ey, ex = start[1], start[0], end[1], end[0]
    #--------------------------------------------- 
    gates = [(sy, sx), (ey, ex)]
    cleaned=Delet_space(cleaned, gates , 200)
    
#---------------------------------------------
    maze = (cleaned // 255).astype(np.uint8)
    sy, sx, ey, ex = start[1], start[0], end[1], end[0]
    if maze[sy, sx] == 0 or maze[ey, ex] == 0:
        maze = 1 - maze
   
    maze2=(maze * 255).astype(np.uint8) 
    cv2.imshow(" Maze array", maze2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
#---------------------------------------------
    # ⚙️ تنظیم پارامترهای بهینه برای ایمنی
    SAFE_MARGIN_PX = 12
    CENTER_WEIGHT = 8.0
    REPULSION_STRENGTH = 15

    # 🚀 اجرای الگوریتم A*
    path, dist = astar_safe(maze, (sy, sx), (ey, ex))
    
    if not path:
        print("❌ No path found.")
        return

    # 🔄 صاف‌کردن و فشرده‌سازی مسیر
    path = smooth_path(path, window=SMOOTH_WINDOW)
    path_comp = compress_straight_segments(path)

    # 🧮 تبدیل مسیر به مختصات ربات
    dobot_path = [image_to_robot(x, y) for (y, x) in path_comp]

    # 💾 ذخیره مسیر ایمن
    with open("dobot_path_safe.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["X_mm", "Y_mm", "Z_mm"])
        writer.writerows(dobot_path)
    print("✅ Saved safe path to dobot_path_safe.csv")

    # 🔴 نمایش مسیر روی تصویر
    solved = warped.copy()
    for (y, x) in path:
        cv2.circle(solved, (x, y), 1, (0, 0, 255), -1)
    cv2.imshow("Solved Maze (Safe Path)", solved)
    getkey()

    # 🎮 اجرای مسیر زنده با ربات
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("⚠️ No Dobot connected — skipping live execution.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    try:
        for i, (x, y, z) in enumerate(dobot_path):
            device.move_to(x, y, Z_Safe, r=0)
            py, px = path_comp[i]
            cv2.circle(solved, (px, py), 4, (0, 255, 255), -1)
            cv2.imshow("Live Path Follow", solved)
            if cv2.waitKey(1) == 27:
                break
            time.sleep(0.05)
        device.move_to(*HOME)
        device.close()
        print("✅ Path executed safely!")
    except Exception as e:
        print(f"⚠️ Error executing path: {e}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()

# ======= بارگذاری کلیدهای API =======
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# ======= گفتار به صدا (ElevenLabs) =======
def speak(text1):
    global Flag_Voice_Text

    print(f"🤖 Speaking: {text1}")
    
    if Flag_Voice_Text==1 :
        audio = elevenlabs.text_to_speech.convert(
        text=text1,
        voice_id="pqHfZKP75CvOlQylNhV4",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",)
        play(audio)
    else  :
        print(text1)

# ======= گفتار به متن (Whisper) =======
def listen_whisper():
  global Flag_Voice_Text
  
  if Flag_Voice_Text==1 :
    print("🎧 Listening... Speak now.")

    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        write(tmpfile.name, SAMPLE_RATE, audio)
        tmp_path = tmpfile.name

    with open(tmp_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )

    text = transcript.text.strip()
    print(f"👤 You said: {text}")
  else :
    text=""
    print("🎧 Pleas Enter Your Command ? ")
    text=input()
  return text.lower()

# ======= تابع تشخیص فرمان =======
def detect_command(text):
    """
    بررسی می‌کند آیا کاربر کلمه play، red، یا green را گفته است
    """
    text = text.lower().strip()

    patterns = {
        "green": [r"\bgreen\b", "سبز"],
        "red": [r"\bred\b", "قرمز"],
        "play": [r"\bplay\b", "پلی", "شروع"],
        "exit" :[r"\bexit\b"],
        "algorithm" :[r"\bexit\b"]
    }

    for key, words in patterns.items():
        for w in words:
            if re.search(w, text):
                return key

    return None

# ======= حالت‌ها =======
def Greeting():
    global TextGlobal, State
    print("🧩 Greeting")
    speak("Welcome! I am your maze assistant. how can i help you")
    State = 2

def Record1():
    global TextGlobal, State
    print("🎙️ Recording user command...")
    TextGlobal = listen_whisper()
    State = 3

# ✅ نسخه جدید تابع Check_Word با تحلیل متن
def Chek_Word():
    global TextGlobal, State, start_color
    print("🧠 Checking words with GPT intent detection...")

    if not TextGlobal:
        speak("I didn’t hear you. Please say play or choose a color.")
        State = 2
        return

    try:
        # 💬 مرحله ۱: فرستادن جمله‌ی کاربر به ChatGPT برای تشخیص هدف
        prompt = f"""
        You are an intent classification assistant.
        The user said: "{TextGlobal}"

        Your task is to classify the user's intent into one of these actions:
        [play, stop, explain, red, green, algorithm, exit, none]

        - "play" → user wants to play the maze.
        - "stop" → user doesn't want to play.
        - "explain" → user wants you to explain something about the game.
        - "red"/"green" → user chose a starting color.
        - "algorithm" → user asked about A* or the maze-solving method.
        - "exit" → user wants to end the presentation.
        - "none" → uncertain intent.

        Respond **only with one word** from the list above.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )

        intent = response.choices[0].message.content.strip().lower()
        print(f"🎯 GPT detected intent: {intent}")

        # 💡 مرحله ۲: تصمیم‌گیری بر اساس intent
        if intent == "play":
            #speak("You want to play. Please choose a color: red or green.")
            State = 6

        elif intent == "red":
            speak("You chose red. Starting from the red point.")
            start_color = "red"
            State = 7

        elif intent == "green":
            speak("You chose green. Starting from the green point.")
            start_color = "green"
            State = 7

        elif intent == "algorithm":
            speak("You asked about the algorithm. Let me explain the A-star pathfinding method.")
            TextGlobal = "Explain how the A* algorithm solves the maze."
            State = 4  # برو برای دریافت توضیح از GPT

        elif intent == "explain":
            speak("Sure, I will explain how this maze system works.")
            TextGlobal = "Explain how this maze system works."
            State = 4

        elif intent == "stop":
            speak("Okay, we will not play right now.")
            State = 1  # برگرد به حالت اول یا آماده‌به‌کار

        elif intent == "exit":
           # speak("Presentation finished. Thank you for listening!")
            State = 8

        else:
           # speak("I didn’t clearly understand. Could you please repeat?")
            State = 4  # برگرد به ضبط مجدد

    except Exception as e:
        print(f"❌ Error in Chek_Word: {e}")
        #speak("Sorry, I had trouble understanding. Please try again.")
        State = 6

#-----------------------------------------------------------------------
def Get_Answer():
    global TextGlobal, State
    print("🧩 Getting answer from ChatGPT...")

    if not TextGlobal or TextGlobal=="you" :
        speak("I didn’t receive any text yet. Please say something first.")
        State = 2
        return

    try:
        # پیام به ChatGPT
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI maze assistant who explains clearly and briefly."},
                {"role": "user", "content": TextGlobal},
            ],
        )

        # دریافت پاسخ و ذخیره در TextGlobal
        answer = response.choices[0].message.content.strip()
        TextGlobal = answer[:200]

        print(f"🤖 GPT Response: {answer}")
        #speak("Here’s my answer for you.")
        #speak(answer)

        # رفتن به حالت بعدی
        State = 5

    except Exception as e:
        print(f"❌ Error in Get_Answer: {e}")
        speak("I encountered an error while contacting ChatGPT.")
        State = 2


def Play_Answer():
    global TextGlobal, State
    speak("Here’s my answer for you.")
    speak(TextGlobal)
    State = 6  # برگرد به شروع یا ادامه

def Play_Want():
    global TextGlobal, State
    print("🧩 Play_Want")
    speak("You wanted to play. Please say the color to start the maze: red or green.")
    State = 2

   
def StartMaze():
    global TextGlobal, State
    print("🧩 Play_Want")
    speak("Start Play")
    solve_maze_and_get_path(run_robot=True)
    State = 6

def EndPlay():
    global TextGlobal, State
    print("🧩 EndPlay")
    speak("Presentation finished. Thank you for listening! bye Have a good one")
    State = 1000
# ======= گراف حالت‌ها =======
Machine_State_Graph = {
    1: {"Agent": Greeting},
    2: {"Agent": Record1},
    3: {"Agent": Chek_Word},
    4: {"Agent": Get_Answer},
    5: {"Agent": Play_Answer},
    6: {"Agent": Play_Want},
    7: {"Agent": StartMaze},
    8: {"Agent": EndPlay},
}

# ======= برنامه اصلی =======
def main():
    global State,Flag_Exit
  
    while not Flag_Exit:
        if State in Machine_State_Graph:
            print(f"\n▶️ -----------------------Running state {State}")
           
            Machine_State_Graph[State]["Agent"]()
          
            time.sleep(1)
        else:
            Flag_Exit = 1

    #speak("Presentation finished. Thank you for listening!")

# ======= اجرای برنامه =======
if __name__ == "__main__":
    main()

# =================== Run ===================
#if __name__ == "__main__":
 #   solve_maze_and_get_path(run_robot=True)
