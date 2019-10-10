import flow_utils as flut

cap = flut.get_cv_capture(10)
while not flut.wait_key():
    flow_amt = flut.get_flow_amount(cap, resize_to=(64,64), frame_count=20)
    print(flow_amt)

