To do list:
- Rendi il codice portable (i.e. in alcune linee di codice l'eseguibile di gioco  
e' inserito come indirizzo assoluto e non come indirizzo relativo, e non riesco a farlo  
funzionare in altro modo. Se riesci a risolvere questo bug e' tanto)  
- Verifica tramite **stable_baseline3** (SubprocVecEnv) la possibilita' di avere  
piu' istanze di IKEMEN GO aperte *(servira' usare -port o altre impostazioni di IKEMEN*  
*probabilmente)*  

Next up:
- MLP expansion e training del teacher MLP tramite self play PPO  
- Eventualmente introdurre un RNN per rendere il modello piu' robusto  
- impostazione e training vision student tramite self play PPO e distillation  


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