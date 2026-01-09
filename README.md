To do list:
- Verify that the sceduling logic of the send/recieve is actually reasonable, we may want to recieve the action first
- Make a better protocol with reset (ack and confirmation), pause (+ folloup)
- Try to automate the ikemen section of the code so that args can be decided at runtime
- Revision of the controls protocol based in the network head shape


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