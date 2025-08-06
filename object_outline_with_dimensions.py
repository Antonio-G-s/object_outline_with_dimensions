"""Extract outlines of objects in images using a coin for scale."""

from typing import Optional, Tuple

import cv2
import numpy as np
import os
import shutil
from tkinter import HORIZONTAL, Button, Canvas, Frame, Label, Scale, Tk, Toplevel
from PIL import Image, ImageTk

# === CONFIG ===
REFERENCE_DIAMETER_MM = 23.25
INPUT_FOLDER = "images"
OUTPUT_FOLDER = "output"
RETRY_FOLDER = "retry"
os.makedirs(RETRY_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def resize_for_display(
    img: np.ndarray, max_height: int = 900, max_width: int = 1600
) -> np.ndarray:
    """Resize ``img`` to fit within ``max_height`` and ``max_width``."""
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    return cv2.resize(img, (int(w * scale), int(h * scale)))

def detect_coin_scale(
    image: np.ndarray,
    lower_thresh: int = 15,
    upper_thresh: int = 45,
    min_area: int = 100,
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[np.ndarray]]:
    """Detect a yellow coin and return its center, radius, and contour.

    ``bbox`` restricts the search to ``(x1, y1, x2, y2)`` in ``image``.
    """
    offset_x, offset_y = 0, 0
    roi = image
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([lower_thresh, 50, 50])
    upper_yellow = np.array([upper_thresh, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_circularity = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        circ = 4 * np.pi * (area / (peri**2))
        if 0.3 < circ < 1.3 and area > min_area and circ > best_circularity:
            best = cnt
            best_circularity = circ

    if best is None:
        return None, None, None, None

    (x, y), r = cv2.minEnclosingCircle(best)
    x, y, r = int(x) + offset_x, int(y) + offset_y, int(r)
    best = best + np.array([[[offset_x, offset_y]]])
    return x, y, r, best

def extract_object_contour(
    image: np.ndarray,
    coin_x: Optional[int],
    coin_y: Optional[int],
    blur_strength: int = 5,
    threshold_val: int = 0,
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[np.ndarray]:
    """Return the largest contour not overlapping the coin.

    ``bbox`` restricts detection to ``(x1, y1, x2, y2)`` in ``image``.
    """
    offset_x, offset_y = 0, 0
    roi = image
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
        if coin_x is not None and coin_y is not None:
            coin_x -= offset_x
            coin_y -= offset_y

    if blur_strength % 2 == 0:
        blur_strength += 1
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, blur_strength)
    _, thresh = cv2.threshold(
        blurred,
        threshold_val,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU if threshold_val == 0 else cv2.THRESH_BINARY_INV,
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if coin_x is not None and coin_y is not None:
        contours = [
            cnt
            for cnt in contours
            if cv2.pointPolygonTest(cnt, (float(coin_x), float(coin_y)), False) < 0
        ]
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    best = best + np.array([[[offset_x, offset_y]]])
    return best

def show_verification_gui(
    original_image: np.ndarray,
    object_contour: np.ndarray,
    coin_contour: np.ndarray,
    image_name: str,
    mm_per_pixel: float,
) -> None:
    """Display the detected outline and ask the user to verify it."""
    window = Tk()
    window.title("Is the outline correct?")
    main_frame = Frame(window)
    main_frame.pack()

    coin_cnt = coin_contour
    obj_cnt = object_contour
    mmpp = mm_per_pixel

    display = resize_for_display(original_image)
    scale_x = display.shape[1] / original_image.shape[1]
    scale_y = display.shape[0] / original_image.shape[0]

    def redraw_main_canvas() -> None:
        nonlocal tk_img_main
        verify_img = display.copy()
        if coin_cnt is not None:
            coin_scaled = np.array(
                [[[int(pt[0][0] * scale_x), int(pt[0][1] * scale_y)]] for pt in coin_cnt]
            )
            cv2.drawContours(verify_img, [coin_scaled], -1, (0, 255, 255), 2)
        if obj_cnt is not None:
            obj_scaled = np.array(
                [[[int(pt[0][0] * scale_x), int(pt[0][1] * scale_y)]] for pt in obj_cnt]
            )
            cv2.drawContours(verify_img, [obj_scaled], -1, (0, 0, 255), 2)
        img_pil = Image.fromarray(cv2.cvtColor(verify_img, cv2.COLOR_BGR2RGB))
        tk_img_main = ImageTk.PhotoImage(img_pil)
        canvas_main.itemconfig(canvas_img_main, image=tk_img_main)

    tk_img_main = None  # placeholder for PhotoImage reference
    verify_img_init = display.copy()
    if coin_cnt is not None:
        coin_scaled = np.array(
            [[[int(pt[0][0] * scale_x), int(pt[0][1] * scale_y)]] for pt in coin_cnt]
        )
        cv2.drawContours(verify_img_init, [coin_scaled], -1, (0, 255, 255), 2)
    if obj_cnt is not None:
        obj_scaled = np.array(
            [[[int(pt[0][0] * scale_x), int(pt[0][1] * scale_y)]] for pt in obj_cnt]
        )
        cv2.drawContours(verify_img_init, [obj_scaled], -1, (0, 0, 255), 2)
    tk_img_main = ImageTk.PhotoImage(
        Image.fromarray(cv2.cvtColor(verify_img_init, cv2.COLOR_BGR2RGB))
    )
    canvas_main = Canvas(main_frame, width=verify_img_init.shape[1], height=verify_img_init.shape[0])
    canvas_img_main = canvas_main.create_image(0, 0, anchor="nw", image=tk_img_main)
    canvas_main.pack()

    def on_yes() -> None:
        folder = os.path.join(OUTPUT_FOLDER, os.path.splitext(image_name)[0])
        os.makedirs(folder, exist_ok=True)
        shutil.copy2(os.path.join(INPUT_FOLDER, image_name), os.path.join(folder, "original.jpg"))

        white = np.ones_like(original_image) * 255
        cv2.drawContours(white, [obj_cnt], -1, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(folder, "verify.jpg"), white)

        poly = cv2.approxPolyDP(obj_cnt, 0.01 * cv2.arcLength(obj_cnt, True), True)
        outline = np.ones_like(original_image) * 255
        cv2.polylines(outline, [poly], isClosed=True, color=(0, 0, 0), thickness=3)

        for i in range(len(poly)):
            pt1 = tuple(poly[i][0])
            pt2 = tuple(poly[(i + 1) % len(poly)][0])
            mid_x = int((pt1[0] + pt2[0]) / 2)
            mid_y = int((pt1[1] + pt2[1]) / 2)
            dist = np.linalg.norm(np.array(pt1) - np.array(pt2)) * mmpp
            cv2.putText(
                outline,
                f"{dist:.1f} mm",
                (mid_x, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )

        cv2.imwrite(os.path.join(folder, "final_outline.jpg"), outline)
        window.destroy()

    def on_no() -> None:
        shutil.move(os.path.join(INPUT_FOLDER, image_name), os.path.join(RETRY_FOLDER, image_name))
        window.destroy()

    def on_modify() -> None:
        nonlocal coin_cnt, obj_cnt, mmpp

        modify_win = Toplevel(window)
        modify_win.title("Modify Detection")

        disp = resize_for_display(original_image)
        disp_scale_x = original_image.shape[1] / disp.shape[1]
        disp_scale_y = original_image.shape[0] / disp.shape[0]

        canvas = Canvas(modify_win, width=disp.shape[1], height=disp.shape[0])
        canvas.pack()

        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        canvas_img = canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.image = photo

        coin_bbox: Optional[Tuple[int, int, int, int]] = None
        object_bbox: Optional[Tuple[int, int, int, int]] = None
        current_coin: Optional[np.ndarray] = coin_cnt
        current_obj: Optional[np.ndarray] = obj_cnt
        current_mmpp = mmpp

        start_x = start_y = 0
        temp_rect = None
        select_mode = None

        def begin_select(event):
            nonlocal start_x, start_y, temp_rect
            if select_mode is None:
                return
            start_x, start_y = event.x, event.y
            temp_rect = canvas.create_rectangle(
                start_x, start_y, start_x, start_y, outline="lime", width=2, tags="temp"
            )

        def update_select(event):
            if temp_rect is not None:
                canvas.coords(temp_rect, start_x, start_y, event.x, event.y)

        def end_select(event):
            nonlocal coin_bbox, object_bbox, temp_rect
            if select_mode is None:
                return
            end_x, end_y = event.x, event.y
            x1, x2 = sorted([start_x, end_x])
            y1, y2 = sorted([start_y, end_y])
            if select_mode == "coin":
                canvas.delete("coin_box")
                canvas.create_rectangle(x1, y1, x2, y2, outline="yellow", width=2, tags="coin_box")
                coin_bbox = (
                    int(x1 * disp_scale_x),
                    int(y1 * disp_scale_y),
                    int(x2 * disp_scale_x),
                    int(y2 * disp_scale_y),
                )
            else:
                canvas.delete("obj_box")
                canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="obj_box")
                object_bbox = (
                    int(x1 * disp_scale_x),
                    int(y1 * disp_scale_y),
                    int(x2 * disp_scale_x),
                    int(y2 * disp_scale_y),
                )
            if temp_rect is not None:
                canvas.delete(temp_rect)
                temp_rect = None
            update_preview()

        canvas.bind("<ButtonPress-1>", begin_select)
        canvas.bind("<B1-Motion>", update_select)
        canvas.bind("<ButtonRelease-1>", end_select)

        def enable_coin():
            nonlocal select_mode
            select_mode = "coin"

        def enable_object():
            nonlocal select_mode
            select_mode = "object"

        coin_lower = Scale(modify_win, from_=0, to=60, orient=HORIZONTAL, label="Lower Hue", command=lambda v: update_preview())
        coin_lower.set(15)
        coin_lower.pack()
        coin_upper = Scale(modify_win, from_=0, to=120, orient=HORIZONTAL, label="Upper Hue", command=lambda v: update_preview())
        coin_upper.set(45)
        coin_upper.pack()
        coin_area = Scale(modify_win, from_=50, to=1000, orient=HORIZONTAL, label="Min Area", command=lambda v: update_preview())
        coin_area.set(100)
        coin_area.pack()

        obj_blur = Scale(modify_win, from_=1, to=21, orient=HORIZONTAL, label="Blur Strength", command=lambda v: update_preview(), resolution=2)
        obj_blur.set(5)
        obj_blur.pack()
        obj_thresh = Scale(modify_win, from_=0, to=255, orient=HORIZONTAL, label="Threshold Sensitivity", command=lambda v: update_preview())
        obj_thresh.set(0)
        obj_thresh.pack()

        preview_label = Label(modify_win)
        preview_label.pack()

        preview_photo = None

        def update_preview() -> None:
            nonlocal current_coin, current_obj, current_mmpp, preview_photo
            low = coin_lower.get()
            high = coin_upper.get()
            area_min = coin_area.get()
            blur_val = obj_blur.get()
            thresh_val = obj_thresh.get()

            coin_x, coin_y, coin_r, coin_tmp = detect_coin_scale(
                original_image, low, high, area_min, coin_bbox
            )
            obj_tmp = extract_object_contour(
                original_image, coin_x, coin_y, blur_val, thresh_val, object_bbox
            )
            preview_img = original_image.copy()
            if coin_tmp is not None:
                cv2.drawContours(preview_img, [coin_tmp], -1, (0, 255, 255), 2)
            if obj_tmp is not None:
                cv2.drawContours(preview_img, [obj_tmp], -1, (0, 0, 255), 2)
            preview_disp = resize_for_display(preview_img)
            preview_photo = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(preview_disp, cv2.COLOR_BGR2RGB))
            )
            preview_label.configure(image=preview_photo)
            preview_label.image = preview_photo

            current_coin = coin_tmp
            current_obj = obj_tmp
            if coin_r is not None:
                current_mmpp = REFERENCE_DIAMETER_MM / (2 * coin_r)

        update_preview()

        def confirm() -> None:
            nonlocal coin_cnt, obj_cnt, mmpp
            if current_coin is not None and current_obj is not None:
                coin_cnt = current_coin
                obj_cnt = current_obj
                mmpp = current_mmpp
                redraw_main_canvas()
            modify_win.destroy()

        def discard() -> None:
            modify_win.destroy()

        Button(modify_win, text="Select Coin Area", command=enable_coin).pack(pady=5)
        Button(modify_win, text="Select Object Area", command=enable_object).pack(pady=5)
        Button(modify_win, text="Confirm", command=confirm).pack(side="left", padx=50, pady=10)
        Button(modify_win, text="Discard", command=discard).pack(side="right", padx=50, pady=10)

    Button(
        window,
        text="YES",
        command=on_yes,
        height=2,
        width=20,
        bg="green",
        fg="white",
    ).pack(side="left", padx=30, pady=10)
    Button(
        window,
        text="NO",
        command=on_no,
        height=2,
        width=20,
        bg="red",
        fg="white",
    ).pack(side="right", padx=30, pady=10)
    Button(
        window,
        text="MODIFY",
        command=on_modify,
        height=2,
        width=20,
        bg="orange",
        fg="black",
    ).pack(side="bottom", pady=10)
    window.mainloop()

def run_image(img_name: str) -> None:
    """Process a single image and display results."""
    full_path = os.path.join(INPUT_FOLDER, img_name)
    image = cv2.imread(full_path)
    if image is None:
        return
    coin_x, coin_y, coin_r, coin_contour = detect_coin_scale(image)
    if coin_x is None:
        print(f"Skipping {img_name}: coin not found.")
        shutil.move(full_path, os.path.join(RETRY_FOLDER, img_name))
        return

    mm_per_pixel = REFERENCE_DIAMETER_MM / (2 * coin_r)
    object_contour = extract_object_contour(image, coin_x, coin_y)
    if object_contour is None:
        print(f"Skipping {img_name}: object not found.")
        shutil.move(full_path, os.path.join(RETRY_FOLDER, img_name))
        return

    show_verification_gui(image, object_contour, coin_contour, img_name, mm_per_pixel)


def main() -> None:
    """Run the outline extraction on all images in ``INPUT_FOLDER``."""
    for img_name in os.listdir(INPUT_FOLDER):
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            run_image(img_name)


if __name__ == "__main__":
    main()
