package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"sync"
	"time"
)

type RLGameState struct {
	GameTick   int     `json:"tick"`
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

type RLMessage struct {
	State  RLGameState `json:"state"`
	FrameW int         `json:"frame_w,omitempty"`
	FrameH int         `json:"frame_h,omitempty"`
}

type AgentAction struct {
	P1Move string `json:"p1_move"`
	P1Btn  string `json:"p1_btn"`
	P2Move string `json:"p2_move"`
	P2Btn  string `json:"p2_btn"`
	Reset  bool   `json:"reset"`
}

var (
	conn        net.Conn
	listener    net.Listener
	mu          sync.Mutex
	IsConnected bool = false
	netReader   *bufio.Reader
)

func writeU32(conn net.Conn, v uint32) error {
	var buf [4]byte
	binary.BigEndian.PutUint32(buf[:], v)
	_, err := conn.Write(buf[:])
	return err
}

func StartRLServer() {
	port := ":" + sys.cmdFlags["-port"]
	fmt.Println(sys.cmdFlags)

	fmt.Println("--- RL SERVER: Avvio RAW su porta " + port + " ---")
	ln, err := net.Listen("tcp", port)
	if err != nil {
		fmt.Println("Errore bind:", err)
		return
	}
	listener = ln

	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				continue
			}

			mu.Lock()
			if conn != nil {
				conn.Close()
			}
			conn = c
			// TCP No Delay non è accessibile facilmente da net.Conn standard, ma bufio aiuta
			netReader = bufio.NewReader(conn)
			IsConnected = true
			mu.Unlock()

			fmt.Println("--- PYTHON CONNESSO (RAW MODE) ---")
		}
	}()
}

func disconnect() AgentAction {
	fmt.Println("RL connection lost")
	conn.Close()
	IsConnected = false
	return AgentAction{}
}

func readAction(conn net.Conn) (AgentAction, error) {
	var size uint32
	if err := binary.Read(conn, binary.BigEndian, &size); err != nil {
		return AgentAction{}, err
	}

	buf := make([]byte, size)
	if _, err := io.ReadFull(conn, buf); err != nil {
		return AgentAction{}, err
	}

	var action AgentAction
	err := json.Unmarshal(buf, &action)
	return action, err
}

func SyncWithPython(state RLGameState, frame []byte, w, h int) AgentAction {
	mu.Lock()
	defer mu.Unlock()

	fmt.Println("Sync with python")
	if !IsConnected || conn == nil {
		return AgentAction{}
	}

	var (
		action AgentAction
		msg    RLMessage
	)

	msg.FrameH = h
	msg.FrameW = w
	msg.State = state

	stateJSON, err := json.Marshal(msg)
	if err != nil {
		return AgentAction{}
	}
	fmt.Println("Set Write Deadline")
	conn.SetWriteDeadline(time.Now().Add(2 * time.Second))

	// ---- SEND STATE ----
	fmt.Println("First write")
	if err := writeU32(conn, uint32(len(stateJSON))); err != nil {
		return disconnect()
	}
	fmt.Println("Second write")
	if _, err := conn.Write(stateJSON); err != nil {
		return disconnect()
	}

	// ---- SEND IMAGE ----
	fmt.Println("Third write")
	if err := writeU32(conn, uint32(len(frame))); err != nil {
		return disconnect()
	}
	fmt.Println("Fourth write")
	if _, err := conn.Write(frame); err != nil {
		return disconnect()
	}

	// ---- READ ACTION ----
	fmt.Println("Read")
	conn.SetReadDeadline(time.Now().Add(5 * time.Second))
	action, err = readAction(conn)
	if err != nil {
		return disconnect()
	}

	return action
}
