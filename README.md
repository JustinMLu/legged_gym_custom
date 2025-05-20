# ARCAD Gym  #

<img src="https://github.com/user-attachments/assets/1a5062e2-f062-4528-929f-09ea0b1da057" width="400"/>
<img src="https://github.com/user-attachments/assets/aafa9d2f-bfbf-452e-8215-3e1fd6227e85" width="400"/>

---

ARCAD Gym is an in-house extension of legged_gym for the Agile Robotics Control and Design (ARCAD) Group, at the University of Michigan. This fork adds a sim-to-sim-to-real inference pipeline from Isaac Sim to Mujoco and then onto the real robot and completely reimplements Regularized Online Adaptation (ROA). Bugfixes to the terrain generation and base RL pipeline have also been made.


Custom functions implemented:
- Regularized Online Adaptation (ROA) w/ MLP state estimator
- Completely modular observation buffers in Isaac Gym - no more slicing a giant `self.obs_buf` tensor over and over
- Fully-functional Mujoco and Go2 robot deployment scripts
- Xbox controller support for Mujoco and Isaac Sim 
- The code is readable by a human being

**Affiliation**: Justin Lu - ARCAD Lab, University of Michigan

---

### 🚧 Installation ###
1. Clone this repository
2. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
3. Install pytorch 1.10 with cuda-11.3:
    - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
4. Install Isaac Gym
   - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`
5. Install the version of rsl_rl that comes with this repository
   -  `cd arcad_gym/rsl_rl && pip install -e .` 
6. Install arcad_gym
   - `cd arcad_gym && pip install -e .`

---

### 💾 Export Network (with ROA) ###
Playing a policy automatically exports its network components to `logs/{experiment_name}/exported/policies`. Exported files include:
- The policy (MLP only, RNN support is deprecated) is exported as ```policy.pt```
- The estimator network is exported as ```estimator.pt```
- The adaptation module (ROA) is exported as ```adaptation_moduke.pt```
- The scan encoder is exported as ```scan_encoder.pt```

---

### 🖥️ Training a Parkour Policy ###
Currently, ```ARCAD Gym``` is optimized for quadrupedal robot parkour. Follow these steps to train and deploy a policy for the unitree Go2 that can eprform complex maneuvers like hurdles and jumps on & over objects.

1. Train the base parkour policy:  
  ```python legged_gym/scripts/train.py --task=go2_parkour --headless```

2. Finetune the base parkour policy:  
  ```python legged_gym/scripts/train.py --task=go2_parkour_finetune --headless```

3. Generate scan observations:
    - Uncomment lines 541-558 in ```play.py```
    - Run the script to generate ```FAKE_SCAN_OBS.txt```
    - Rename this file and move it to ```deploy/base```
    - Reference examples: ```SCAN_v12_ft_i.txt``` and ```SCAN_v12_ft_iii.txt``` (included for pre-trained policies)

4. Organize policy files:
    - Create a directory in ```deploy/networks/go2/<your_policy_name>```
    - Move the exported policy files into this directory

5. Configure deployment:
    - Update the ```model_name``` in ```deploy/configs/go2.yaml``` to match your policy folder
    - Ensure parameters match your IsaacGym configuration
    - Update the filename in ```deploy_base-deploy_base.py``` to point to your scan observations file

6. Deploy in Mujoco:  
  ```python deploy/deploy_mujoco/deploy_mujoco.py go2.yaml```
    -  Plug in an Xbox controller before running and control the robot! 

7. Deploy to physical robot:  
  ```python deploy/deploy_real/deploy_real.py eth0 go2.yaml```
    -  Connect to your Go2 robot via ethernet cable or SSH into its onboard Jetson
    - Follow the setup instructions in [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) before attempting real-world deployment

---

### 🖥️ General Usage ###
For standard (non-parkour) training and evaluation:

1. Train:  
  ```python legged_gym/scripts/train.py --task=<your_task_here>```
    -  To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
    -  To run headless (no rendering) add `--headless`.
    - **Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
    - The trained policy is saved in `issacgym_anymal/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
    - E.g: loading a specific run (Feb19_19-10-10_goober) at a specific checkpoint (700) and resuming headless training: 
        - ```python legged_gym/scripts/train.py --task=go2 --load_run=Feb19_19-10-10_goober --checkpoint=700 --headless --resume```
    
    -  The following command line arguments override the values set in the config files:
        - ```--task=TASK```: Task name.
        - ```--resume```: Resume training from a checkpoint
        - ```--experiment_name=EXPERIMENT_NAME```: Name of the experiment to run or load.
        - ```--run_name=RUN_NAME```: Name of the run to load during playback (I think...)
        - ```--load_run=LOAD_RUN```: Name of the run to load during training when resume=True. If -1: will load the last run.
        - ```--checkpoint=CHECKPOINT```:  Saved model checkpoint number. If -1: will load the last checkpoint.
        - ```--num_envs=NUM_ENVS```:  Number of environments to create.
        - ```--seed=SEED```:  Random seed.
        - ```--max_iterations=MAX_ITERATIONS```:  Maximum number of training iterations.

2. Play a trained policy:  
```python legged_gym/scripts/play.py --task=<your_task_here>```
    - By default, the loaded policy is the last model of the last run of the experiment folder.
    - Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.

2. Play a trained policy and control it with your Xbox controller in Isaac Gym:  
```python legged_gym/scripts/control_and_play.py --task=<your_task_here>```
    - You can edit the camera settings by directly editing ```control_and_play.py```
    - Spawning more than one robot is still supported

---

### ⚠️ Troubleshooting ###
1. If you get the following error: `ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`, do: `sudo apt install libpython3.8`. It is also possible that you need to do `export LD_LIBRARY_PATH=/path/to/libpython/directory` / `export LD_LIBRARY_PATH=/path/to/conda/envs/your_env/lib`(for conda user. Replace /path/to/ to the corresponding path.).

---

### ☢️ Known Issues ###
1. The contact forces reported by `net_contact_force_tensor` are unreliable when simulating on GPU with a triangle mesh terrain. A workaround is to use force sensors, but the force are propagated through the sensors of consecutive bodies resulting in an undesirable behaviour. However, for a legged robot it is possible to add sensors to the feet/end effector only and get the expected results. When using the force sensors make sure to exclude gravity from the reported forces with `sensor_options.enable_forward_dynamics_forces`. Example:
```
    sensor_pose = gymapi.Transform()
    for name in feet_names:
        sensor_options = gymapi.ForceSensorProperties()
        sensor_options.enable_forward_dynamics_forces = False # for example gravity
        sensor_options.enable_constraint_solver_forces = True # for example contacts
        sensor_options.use_world_frame = True # report forces in world frame (easier to get vertical components)
        index = self.gym.find_asset_rigid_body_index(robot_asset, name)
        self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)
    (...)

    sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
    self.gym.refresh_force_sensor_tensor(self.sim)
    force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
    self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]
    (...)

    self.gym.refresh_force_sensor_tensor(self.sim)
    contact = self.sensor_forces[:, :, 2] > 1.
```
