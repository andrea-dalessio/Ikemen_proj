from pycode import StudentModel, TeacherModel, IkemenEnvironment, SuperEnvironment
import os
import torch
import argparse

parser = argparse.ArgumentParser(prog="main.py", description="Select your running mode!", usage="%(prog)s [options]")
parser.add_argument("-n", help="Chose how many environments to use", action="store")
parser.add_argument('--teacherTrain', action='store_true', help='Run training with teacher model')
parser.add_argument('--teacherEval', action='store_true', help='Evaluate teacher model')
parser.add_argument('--studentTrain', action='store_true', help='Run training with student model')
parser.add_argument('--eval', action='store_true', help='Run evaluation with trained student model')
parser.add_argument('--headless', action='store_true', help='Disable window')

args = parser.parse_args()

headless = args.headless

dirList = os.listdir()

if 'logs' not in dirList:
    os.mkdir(f"{os.getcwd()}/logs")
    print("Log directory created")
else:
    for log in os.listdir(f"{os.getcwd()}/logs"):
        if not log.endswith('.csv'):
            os.remove(f"{os.getcwd()}/logs/{log}")
    print("Log dirctory OK")

if 'models_saves_t' not in dirList:
    os.mkdir(f"{os.getcwd()}/models_saves_t")
    print("Saves directory created")
else:
    print("Saves dirctory teacher OK")

if 'models_saves_t' not in dirList:
    os.mkdir(f"{os.getcwd()}/models_saves_s")
    print("Saves directory created")
else:
    print("Saves dirctory student OK")

envNum = int(args.n) if args.n is not None else 4
print("Selected {} environment(s)".format(envNum))
if args.teacherTrain:
    env = SuperEnvironment(training_mode="teacher", environment_number=envNum, headless=headless)
    model = TeacherModel(env, load_checkpoint=True)
    try:
        model.trainPPO()
    finally:
        env.disconnect()
        env.close_game()
    
elif args.studentTrain:
    env = SuperEnvironment(training_mode="student", environment_number=envNum, headless=headless)
    model = StudentModel(env, load_checkpoint=True)
    try:
        model.trainPPO()
    finally:
        env.disconnect()
        env.close_game()
        
# elif args.teacherEval:
#     env = IkemenEnvironment(training_mode="teacher", port=8080)
#     model = TeacherModel(env, load_checkpoint=True)
#     done = False
#     env.connect()
#     env.launch_game()
#     try:
#         env.start()
#         first_state_raw, _  = env.wait_for_match_start()
#         env.previousState = first_state_raw
#         first_state = env.normalizeState(first_state_raw)
#     except Exception as e:
#         print(f"Critical Error: Could not connect to environment. {repr(e)}")
#         env.close_game()
    
#     state = torch.tensor(first_state, dtype=torch.float32, device=model.device)
#     while not done:
        
#         action = model.act(state)
        
#         pass


elif args.eval:
    env = IkemenEnvironment(training_mode="student", port=8080)
    model = StudentModel(env)
    ...
else:
    parser.print_help()
    

