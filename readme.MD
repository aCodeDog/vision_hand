
# Install 


## step 1 -  Install Isaac Gym

   - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`)
## step 2 install avp_stream


```
pip install avp_stream
```
### Install the ' Tracking Streamer ' app on Vision Pro

This app is now officially on VisionOS App Store! You can search for Tracking Streamer from the App Store and install the app

 ## step 3 install this pkg

 ```
pip install -e .
```

# run


```

python viz_isaac_hand_gym.py --ip 192.168.123.249 # replace your computer local ip. Make sure your computer and VisionPro are on the same local network.

```

# Acknowledgements
This code builds upon following open-source code-bases. Please visit the URLs to see the respective LICENSES:

1. https://github.com/Improbable-AI/VisionProTeleop
2. https://github.com/dexsuite/dex-retargeting
