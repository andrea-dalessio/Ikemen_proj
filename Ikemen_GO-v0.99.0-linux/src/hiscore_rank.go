package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

type rankingEntry struct {
	Score       int      `json:"score"`
	Time        float64  `json:"time"` // minutes with 2 decimals
	Win         int      `json:"win"`
	Lose        int      `json:"lose"`
	Consecutive int      `json:"consecutive"`
	Name        string   `json:"name"`
	Chars       []string `json:"chars"`
	Tmode       int32    `json:"tmode"`
	AILevel     int32    `json:"ailevel"`
}

func round2(x float64) float64 { return math.Round(x*100) / 100 }

func defKey(p string) string {
	p = strings.ReplaceAll(p, "\\", "/")
	base := filepath.Base(p)
	if ext := filepath.Ext(base); ext != "" {
		base = strings.TrimSuffix(base, ext)
	}
	return strings.ToLower(base)
}

func selectedCharKeys() []string {
	// side 0 (P1) selection; fall back to empty
	var out []string
	for _, slot := range sys.sel.selected[0] {
		cn := slot[0]
		if cn >= 0 && cn < len(sys.sel.charlist) {
			out = append(out, defKey(sys.sel.charlist[cn].def))
		}
	}
	return out
}

type runTally struct {
	timeTicks int32
	winP1     int
	loseP1    int
	maxStreak int
	scoreP1   int
}

func tallyRun() runTally {
	var t runTally
	streak := 0
	lastScore := 0
	for _, m := range sys.statsLog.Matches {
		t.timeTicks += m.MatchTime
		if m.WinSide == 1 {
			t.winP1++
			streak++
		} else if m.WinSide == 2 {
			t.loseP1++
			streak = 0
		}
		if streak > t.maxStreak {
			t.maxStreak = streak
		}
		// TotalScore is already cumulative at match end
		if len(m.TotalScore) == 2 {
			lastScore = int(m.TotalScore[0])
		}
	}
	t.scoreP1 = lastScore
	return t
}

func modeCleared(mode string, matches int) bool {
	switch mode {
	case "survival", "survivalcoop", "netplaysurvivalcoop":
		// P1 wins OR reached configured rounds to win
		if sys.winnerTeam() == 1 {
			return true
		}
		rtw := int(sys.motif.SurvivalResultsScreen.RoundsToWin)
		return matches >= rtw && rtw > 0
	default:
		// Arcade / Timeattack / TeamCoop etc.
		return sys.winnerTeam() == 1
	}
}

func rankingTypeFor(mode string) (string, bool) {
	if sys.motif.HiscoreInfo.Ranking == nil {
		return "", false
	}
	v, ok := sys.motif.HiscoreInfo.Ranking[mode]
	return v, ok
}

// Main entry: computes cleared/placement, writes save/stats.json. Returns (cleared, place).
func computeAndSaveRanking(mode string) (bool, int32) {
	// Tally the just-finished run
	tal := tallyRun()
	cleared := modeCleared(mode, len(sys.statsLog.Matches))

	// Read or create stats file
	data, _ := os.ReadFile("save/stats.json")
	if len(data) == 0 {
		data = []byte(`{}`)
	}

	// Update top-level playtime (minutes)
	curPlay := gjson.GetBytes(data, "playtime").Float()
	curPlay = round2(curPlay + float64(tal.timeTicks)/60.0)
	data, _ = sjson.SetBytes(data, "playtime", curPlay)

	// Ensure mode object
	modeBase := "modes." + mode

	// Mode playtime
	modePlay := gjson.GetBytes(data, modeBase+".playtime").Float()
	modePlay = round2(modePlay + float64(tal.timeTicks)/60.0)
	data, _ = sjson.SetBytes(data, modeBase+".playtime", modePlay)

	// Clear counters
	if cleared {
		clearCount := gjson.GetBytes(data, modeBase+".clear").Int()
		data, _ = sjson.SetBytes(data, modeBase+".clear", clearCount+1)
		// Leader clearcount: selected leader is first picked char
		keys := selectedCharKeys()
		if len(keys) > 0 {
			path := modeBase + ".clearcount." + keys[0]
			cur := gjson.GetBytes(data, path).Int()
			data, _ = sjson.SetBytes(data, path, cur+1)
		}
	} else {
		if !gjson.GetBytes(data, modeBase+".clear").Exists() {
			data, _ = sjson.SetBytes(data, modeBase+".clear", 0)
		}
	}

	// If hiscore ranking isn't defined for this mode, we're done.
	rType, ok := rankingTypeFor(mode)
	if !ok {
		_ = os.WriteFile("save/stats.json", data, 0o644)
		return cleared, 0
	}

	// Ranking exceptions (match Lua behavior)
	if rType == "score" && tal.scoreP1 == 0 {
		_ = os.WriteFile("save/stats.json", data, 0o644)
		return cleared, 0
	}
	if rType == "win" && tal.winP1 == 0 {
		_ = os.WriteFile("save/stats.json", data, 0o644)
		return cleared, 0
	}

	// Build existing entries
	var entries []*rankingEntry
	rankPath := modeBase + ".ranking"
	if arr := gjson.GetBytes(data, rankPath); arr.Exists() && arr.IsArray() {
		arr.ForEach(func(_, v gjson.Result) bool {
			e := &rankingEntry{
				Score:       int(v.Get("score").Int()),
				Time:        v.Get("time").Float(),
				Win:         int(v.Get("win").Int()),
				Lose:        int(v.Get("lose").Int()),
				Consecutive: int(v.Get("consecutive").Int()),
				Name:        v.Get("name").Str,
				Tmode:       int32(v.Get("tmode").Int()),
				AILevel:     int32(v.Get("ailevel").Int()),
			}
			if ca := v.Get("chars"); ca.Exists() && ca.IsArray() {
				ca.ForEach(func(_, c gjson.Result) bool {
					e.Chars = append(e.Chars, c.Str)
					return true
				})
			}
			entries = append(entries, e)
			return true
		})
	}

	// New entry (blank name; Go hiscore screen will update it)
	newE := &rankingEntry{
		Score:       tal.scoreP1,
		Time:        round2(float64(tal.timeTicks) / 60.0),
		Win:         tal.winP1,
		Lose:        tal.loseP1,
		Consecutive: tal.maxStreak,
		Name:        "",
		Chars:       selectedCharKeys(),
		Tmode:       int32(sys.tmode[0]),
		AILevel:     int32(sys.cfg.Options.Difficulty),
	}
	entries = append(entries, newE)

	// Sort by mode
	sort.SliceStable(entries, func(i, j int) bool {
		switch rType {
		case "score":
			// higher score first
			return entries[i].Score > entries[j].Score
		case "time":
			// faster (smaller) time first
			return entries[i].Time < entries[j].Time
		case "win":
			// more wins first; tie-breaker: higher score
			if entries[i].Win != entries[j].Win {
				return entries[i].Win > entries[j].Win
			}
			return entries[i].Score > entries[j].Score
		default:
			return false
		}
	})

	// Compute place (1-based) of our new entry before truncation
	place := int32(0)
	for i, e := range entries {
		if e == newE {
			place = int32(i + 1)
			break
		}
	}

	// Truncate to visible items (if configured)
	visible := int(sys.motif.HiscoreInfo.Window.VisibleItems)
	if visible <= 0 {
		visible = len(entries)
	}
	if len(entries) > visible {
		entries = entries[:visible]
		// If our entry got pushed out, don't trigger name entry
		found := false
		for i, e := range entries {
			if e == newE {
				place = int32(i + 1)
				found = true
				break
			}
		}
		if !found {
			place = 0
		}
	}

	// Serialize back to JSON (preserving other keys)
	buf, _ := json.Marshal(entries)
	data, _ = sjson.SetRawBytes(data, rankPath, buf)
	_ = os.WriteFile("save/stats.json", data, 0o644)

	return cleared, place
}
