package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
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
	End    bool   `json:"end"`
}

var (
	conn        net.Conn
	listener    net.Listener
	mu          sync.Mutex
	IsConnected bool = false
	netReader   *bufio.Reader
)

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

func writeAll(conn net.Conn, data []byte, description string) error {
	n, err := conn.Write(data)
	if err != nil {
		return fmt.Errorf("[Sync] failed to write %s (%d/%d bytes): %w\n", description, n, len(data), err)
	}
	if n != len(data) {
		return fmt.Errorf("[Sync] partial write %s (%d/%d bytes)\n", description, n, len(data))
	}
	log.Printf("[Sync] %s write successful (%d bytes)\n", description, n)
	return nil
}

func uint32ToBytes(n uint32) []byte {
	buf := make([]byte, 4)
	binary.BigEndian.PutUint32(buf, n)
	return buf
}

func SyncWithPython(state RLGameState, frame []byte, w, h int) AgentAction {
	mu.Lock()
	defer mu.Unlock()

	log.Printf("[Sync] Start sync with python Frame: %dx%d, State Tick=%d\n", w, h, state.GameTick)
	if !IsConnected || conn == nil {
		log.Println("[Sync] Connection failed")
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
		log.Println("[Sync] failed to marshal state: ", err)
		return AgentAction{}
	}
	fmt.Println("Set Write Deadline")
	conn.SetWriteDeadline(time.Now().Add(10 * time.Second))

	// ---- SEND STATE ----

	if err := writeAll(conn, uint32ToBytes(uint32(len(stateJSON))), "state length"); err != nil {
		return disconnect()
	}
	if err := writeAll(conn, stateJSON, "state JSON"); err != nil {
		return disconnect()
	}

	if err := writeAll(conn, uint32ToBytes(uint32(len(frame))), "frame length"); err != nil {
		return disconnect()
	}
	if err := writeAll(conn, frame, "frame data"); err != nil {
		return disconnect()
	}

	// ---- READ ACTION ----
	conn.SetReadDeadline(time.Now().Add(10 * time.Second))
	action, err = readAction(conn)
	if err != nil {
		log.Printf("[Sync] Failed to read action: %v\n", err)
		return disconnect()
	}
	log.Printf("[Sync] Successfully synced. Received action: %+v", action)
	return action
}
