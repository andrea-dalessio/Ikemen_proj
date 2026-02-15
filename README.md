# Learning I.K.E.M.En GO with Self-play PPO with distillation  

RL algorithm for learning to play IKEMEN GO with a vision model, helped by a Teacher MLP model.  
To run the script, launch  
```python
python main.py <--options>
```  
The options are  
| Name | Description |
| :--- | :--- |
| --teacherTrain | Starts training mode for teacher.<br>If a checkpoint is present<br>resumes from there. |
| --studentTrain | Trains the student, same as teacher. |
| --eval | Runs one instance of the student model.<br>Uses a previous opponent. |
| --headless | Suppresses game window |
| -n <number> | Choose how many concurren envs<br>should run. | 


## State struct

~~~go
type RLGameState struct {
	GameTick int `json:"tick"`
	FrameW   int `json:"frame_w,omitempty"`
	FrameH   int `json:"frame_h,omitempty"`

	P1_HP      int32   `json:"p1_hp"`
	P1_X       float32 `json:"p1_x"`
	P1_Y       float32 `json:"p1_y"`
	P1_Power   int32   `json:"p1_power"`
	P1_LifeMax int32   `json:"p1_life_max"`
	P1_Facing  float32 `json:"p1_facing"`
	P1_AnimNo  int32   `json:"p1_anim_no"`

	P2_HP      int32   `json:"p2_hp"`
	P2_X       float32 `json:"p2_x"`
	P2_Y       float32 `json:"p2_y"`
	P2_Power   int32   `json:"p2_power"`
	P2_LifeMax int32   `json:"p2_life_max"`
	P2_Facing  float32 `json:"p2_facing"`
	P2_AnimNo  int32   `json:"p2_anim_no"`
}
~~~

## Move struct

~~~json
{
"p1_move": "sting", 
"p1_btn": "string", 
"p2_move": "string", 
"p2_btn": "string", 
"reset": "bool"
}
~~~

