from pycode import IkemenEnvironment
import numpy as np
import cv2

env = IkemenEnvironment()

env.connect()

if not env.connected:
    print("Connection failed")
    exit(1)

fail_cnt = 0

try:
    while True:
        state, image = env.step((1,0),(2,0))
        
    
        rgba = image.astype(np.float32)
        print(rgba.shape)
        print(rgba.max(), rgba.min())
        
        alpha = rgba[:, :, 3:4] / 255.0
        alpha[alpha == 0] = 1.0  # avoid divide-by-zero

        rgb = rgba[:, :, :3] / alpha
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("Ikemen Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
except KeyboardInterrupt:
    print("Manual termination")
    env.disconnect()
finally:
    env.disconnect()
    cv2.destroyAllWindows()