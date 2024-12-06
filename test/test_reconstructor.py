import numpy as np
from three_d_reconstructors.object_reconstructor import ObjectReconstructor

def test_reconstructor():
    reconstructor = ObjectReconstructor()
    frames = []
    
    # Generate test frames of rotating object
    for angle in range(0, 360, 45):
        frame = np.zeros((480, 640, 3))
        x = int(320 + 100 * np.cos(np.radians(angle)))
        frame[240:290, x:x+50] = 255
        frames.append(frame)
    
    points = reconstructor.reconstruct_object(frames)
    print(f"Reconstructed {len(points)} 3D points")