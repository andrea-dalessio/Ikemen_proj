from pycode import StudentModel, TeacherModel, IkemenEnvironment, SuperEnvironment
import os
import argparse

parser = argparse.ArgumentParser(description="Select your running mode! --teacherTrain for training with teacher model, --studentTrain for training with student model, --eval for playing with the trained student model")
parser.add_argument('--teacherTrain', action='store_true', help='Run training with teacher model')
parser.add_argument('--studentTrain', action='store_true', help='Run training with student model')
parser.add_argument('--eval', action='store_true', help='Run evaluation with trained student model')

args = parser.parse_args()

dirList = os.listdir()

if 'logs' not in dirList:
    os.mkdir(f"{os.getcwd()}/logs")
    print("Log directory created")
else:
    print("Log dirctory OK")
if 'models_saves' not in dirList:
    os.mkdir(f"{os.getcwd()}/models_saves")
    print("Saves directory created")
else:
    print("Saves dirctory OK")

# --- Teacher Training Mode ---
if args.teacherTrain:
    env = SuperEnvironment(training_mode="teacher")
    model = TeacherModel(env, load_checkpoint=True)
    try:
        model.trainPPO()
    finally:
        env.disconnect()
        env.close_game()
    
elif args.studentTrain:
    env = IkemenEnvironment(training_mode="student", port=8080)
    model = StudentModel(env)
    ...
    
elif args.eval:
    env = IkemenEnvironment(training_mode="student", port=8080)
    model = StudentModel(env)
    ...

    

