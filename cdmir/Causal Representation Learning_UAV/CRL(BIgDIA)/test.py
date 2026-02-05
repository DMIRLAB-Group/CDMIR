import numpy as np
import airsim

def get_image(client):
    responses = client.simGetImages([airsim.ImageRequest("camera_0", airsim.ImageType.DepthPerspective, True, False)], vehicle_name="Drone_0")
    response = responses[0]
    # Reshape to a 2d array with correct width and height
    depth_img_in_meters = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)
    return depth_img_in_meters 

def run():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True, "Drone_0")
    client.armDisarm(True, "Drone_0")
    while True:
        img_GT = get_image(client)
        size = img_GT.shape
        noise = np.random.normal(scale=0,size=size)
        img_noise = img_GT + noise
        pass

   
                
            
if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt:
        pass