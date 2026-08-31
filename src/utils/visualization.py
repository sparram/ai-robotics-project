import cv2
import numpy as np

def render_frame(env, vehicle, state_real, control_type, step, seed, u_action, obstacles_list, screen_w=608, screen_h=608, scaling=5):
    """Genera y dibuja el frame con la información superpuesta de telemetría y CBF."""
    frame = env.render(
        mode="topdown",
        window=False,
        screen_size=(screen_w, screen_h),
        camera_position=vehicle.position,
        target_agent_heading_up=True,
        scaling=scaling,
        text={
            "seed": seed,
            "step": step,
            "speed_kmh": round(state_real[2] * 3.6, 1),
            "mode": control_type,
            "obs_count": len(obstacles_list) if (control_type == "MPC-CBF" and obstacles_list) else 0,
            "steer": round(u_action[0], 2),
            "accel": round(u_action[1], 2)
        }
    )

    # Vector de control (flecha en el centro del vehículo)
    car_cx, car_cy = screen_w // 2, screen_h // 2
    u_steer, u_acc = u_action[0], u_action[1]
    scale_vec = 45.0
    end_x = int(car_cx - u_steer * scale_vec)
    end_y = int(car_cy - u_acc * scale_vec)
    color_vector = (0, 255, 0) if u_acc >= 0 else (0, 0, 255)

    cv2.arrowedLine(frame, (car_cx, car_cy), (end_x, end_y), color_vector, 3, tipLength=0.35)

    # Círculos de radio CBF sobre obstáculos para MPC-CBF
    if control_type == "MPC-CBF" and obstacles_list:
        v_curr = max(state_real[2], 0.0)
        r_cbf_m = 1.4 + 0.2 * v_curr
        theta = state_real[3]

        for obs in obstacles_list:
            rel_pos = obs[:2] - state_real[:2]
            d_fwd_cam = rel_pos[0] * np.cos(theta) + rel_pos[1] * np.sin(theta)
            d_right_cam = rel_pos[0] * np.sin(theta) - rel_pos[1] * np.cos(theta)

            center_x = int(screen_w / 2 + d_right_cam * scaling)
            center_y = int(screen_h / 2 - d_fwd_cam * scaling)

            cv2.circle(frame, (center_x, center_y), int(r_cbf_m * scaling), (255, 255, 0), 2)

    return frame