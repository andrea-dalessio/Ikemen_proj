# Learning I.K.E.M.En GO with Self-play PPO with distillation  

## Side branch 1  

Affrontando il problema del *termination simplification* per eseguire un workaround sul problema di gestione  
del timeout dei rounds. Training eseguito su round infiniti ma reward shaping eseguito per rendere piu'  
aggressivi i players.  

Struttura attuale modello:  
- **Self-play PPO deep residual MLP teacher** su vectorized features da IKEMEN internal memory.  
- **Self-play PPO Vision CNN student con distillation** su stack di frames. Implementando il metodo realizzato  
in main da Davide.  
 

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

## TODO

- Remove the ticks/max_tiks message in the env
- Fix print message update missmatch
