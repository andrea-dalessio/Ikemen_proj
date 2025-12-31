from pycode import IkemenEnvironment
import numpy as np
import cv2

env = IkemenEnvironment()

env.connect()

if not env.connected:
    print("Connection failed")
    exit(1)

try:
    while True:
        state, image = env.step((1,0),(1,0))

        # Correct conversion
        frame = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        print(
            f"B: {image[:,:,0].mean()}\n",  # B
            f"G: {image[:,:,1].mean()}\n",  # G
            f"R: {image[:,:,2].mean()}\n",  # R
            f"A: {image[:,:,3].mean()}\n",  # A
            sep=''
        )

        cv2.imshow("Ikemen Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    pass
finally:
    env.disconnect()
    cv2.destroyAllWindows()