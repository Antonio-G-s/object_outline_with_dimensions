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
    image: np.ndarray, lower_thresh: int = 15, upper_thresh: int = 45
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[np.ndarray]]:
    """Detect a yellow coin and return its center, radius, and contour."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([lower_thresh, 50, 50])
    upper_yellow = np.array([upper_thresh, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_circularity = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        circ = 4 * np.pi * (area / (peri**2))
        if 0.3 < circ < 1.3 and area > 100 and circ > best_circularity:
            best = cnt
            best_circularity = circ

    if best is None:
        return None, None, None, None
    (x, y), r = cv2.minEnclosingCircle(best)
    return int(x), int(y), int(r), best

def extract_object_contour(
    image: np.ndarray, coin_x: int, coin_y: int, threshold_val: int = 0
) -> Optional[np.ndarray]:
    """Return the largest contour not overlapping the coin."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    _, thresh = cv2.threshold(
        blurred,
        threshold_val,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU if threshold_val == 0 else cv2.THRESH_BINARY_INV,
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [cnt for cnt in contours if cv2.pointPolygonTest(cnt, (float(coin_x), float(coin_y)), False) < 0]
    return max(filtered, key=cv2.contourArea) if filtered else None

def modify_parameters(
    image: np.ndarray,
    coin_contour: Optional[np.ndarray],
    object_contour: Optional[np.ndarray],
    image_name: str,
) -> None:
    """Allow the user to tweak detection parameters via a GUI."""
    top = Toplevel()
    top.title("Modify Detection Parameters")

    canvas = Canvas(top, width=image.shape[1], height=image.shape[0])
    canvas.pack()

    # Initial draw
    display_img = image.copy()
    if coin_contour is not None:
        cv2.drawContours(display_img, [coin_contour], -1, (0, 255, 255), 3)
    if object_contour is not None:
        cv2.drawContours(display_img, [object_contour], -1, (0, 0, 255), 3)

    img_disp = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)))
    img_label = Label(canvas, image=img_disp)
    img_label.image = img_disp
    img_label.pack()

    def update_preview(val=None):
        low = yellow_slider.get()
        high = yellow_slider2.get()
        thresh = object_thresh_slider.get()
        coin_x, coin_y, coin_r, coin_cnt = detect_coin_scale(image, low, high)
        obj_cnt = extract_object_contour(image, coin_x or 0, coin_y or 0, thresh)
        display_img2 = image.copy()
        if coin_cnt is not None:
            cv2.drawContours(display_img2, [coin_cnt], -1, (0, 255, 255), 3)
        if obj_cnt is not None:
            cv2.drawContours(display_img2, [obj_cnt], -1, (0, 0, 255), 3)
        img_disp2 = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(display_img2, cv2.COLOR_BGR2RGB)))
        img_label.configure(image=img_disp2)
        img_label.image = img_disp2

    def confirm():
        top.destroy()

    def discard():
        top.quit()
        top.destroy()

    yellow_slider = Scale(top, from_=0, to=50, orient=HORIZONTAL, label="Coin Yellow Lower HSV", command=update_preview)
    yellow_slider.set(15)
    yellow_slider.pack()
    yellow_slider2 = Scale(top, from_=50, to=100, orient=HORIZONTAL, label="Coin Yellow Upper HSV", command=update_preview)
    yellow_slider2.set(45)
    yellow_slider2.pack()
    object_thresh_slider = Scale(top, from_=0, to=255, orient=HORIZONTAL, label="Object Threshold", command=update_preview)
    object_thresh_slider.set(0)
    object_thresh_slider.pack()

    Button(top, text="Confirm", command=confirm).pack(side="left", padx=50, pady=10)
    Button(top, text="Discard", command=discard).pack(side="right", padx=50, pady=10)
    top.mainloop()

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

    display = resize_for_display(original_image)
    scale_x = display.shape[1] / original_image.shape[1]
    scale_y = display.shape[0] / original_image.shape[0]

    # Draw contours
    verify_img = display.copy()
    if coin_contour is not None:
        coin_scaled = np.array([[[int(pt[0][0] * scale_x), int(pt[0][1] * scale_y)]] for pt in coin_contour])
        cv2.drawContours(verify_img, [coin_scaled], -1, (0, 255, 255), 2)
    if object_contour is not None:
        object_scaled = np.array([[[int(pt[0][0] * scale_x), int(pt[0][1] * scale_y)]] for pt in object_contour])
        cv2.drawContours(verify_img, [object_scaled], -1, (0, 0, 255), 2)

    img_pil = Image.fromarray(cv2.cvtColor(verify_img, cv2.COLOR_BGR2RGB))
    tk_img = ImageTk.PhotoImage(img_pil)
    canvas = Canvas(main_frame, width=verify_img.shape[1], height=verify_img.shape[0])
    canvas.create_image(0, 0, anchor="nw", image=tk_img)
    canvas.pack()

    def on_yes():
        folder = os.path.join(OUTPUT_FOLDER, os.path.splitext(image_name)[0])
        os.makedirs(folder, exist_ok=True)
        shutil.copy2(os.path.join(INPUT_FOLDER, image_name), os.path.join(folder, "original.jpg"))

        # Red outline
        white = np.ones_like(original_image) * 255
        cv2.drawContours(white, [object_contour], -1, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(folder, "verify.jpg"), white)

        # Simplified
        poly = cv2.approxPolyDP(object_contour, 0.01 * cv2.arcLength(object_contour, True), True)
        outline = np.ones_like(original_image) * 255
        cv2.polylines(outline, [poly], isClosed=True, color=(0, 0, 0), thickness=3)

        for i in range(len(poly)):
            pt1 = tuple(poly[i][0])
            pt2 = tuple(poly[(i + 1) % len(poly)][0])
            mid_x = int((pt1[0] + pt2[0]) / 2)
            mid_y = int((pt1[1] + pt2[1]) / 2)
            dist = np.linalg.norm(np.array(pt1) - np.array(pt2)) * mm_per_pixel
            cv2.putText(outline, f"{dist:.1f} mm", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imwrite(os.path.join(folder, "final_outline.jpg"), outline)
        window.destroy()

    def on_no():
        shutil.move(os.path.join(INPUT_FOLDER, image_name), os.path.join(RETRY_FOLDER, image_name))
        window.destroy()

    def on_modify():
        modify_parameters(original_image, coin_contour, object_contour, image_name)

    Button(window, text="YES", command=on_yes, height=2, width=20, bg='green', fg='white').pack(side="left", padx=30, pady=10)
    Button(window, text="NO", command=on_no, height=2, width=20, bg='red', fg='white').pack(side="right", padx=30, pady=10)
    Button(window, text="Modify", command=on_modify, height=2, width=20, bg='blue', fg='white').pack(pady=10)
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
