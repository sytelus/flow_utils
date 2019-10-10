import flow_utils as flut

cap = flut.get_cv_capture(10)
for image in flut.get_flow_image(cap, resize_to=(28,392), frame_count=1000000):
    if flut.show_image(image):
        break