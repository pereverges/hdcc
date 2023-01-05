Steps to execute:

1

python3 main.py voicehd.hdcc
python3 main.py emg.hdcc
python3 main.py mnist.hdcc

2

make

3

// /usr/bin/time -l

./voicehd data/ISOLET/isolet_train_data data/ISOLET/isolet_train_labels data/ISOLET/isolet_test_data data/ISOLET/isolet_test_labels      
./mnist data/MNIST/mnist_train_data data/MNIST/mnist_train_labels data/MNIST/mnist_test_data data/MN
IST/mnist_test_labels
./emg data/EMG_based_hand_gesture/patient_2_train_data data/EMG_based_hand_gesture/patient_2_train_labels data/EMG_based_hand_gesture/patient_2_test_data data/EMG_based_hand_gesture/patient_2_test_labels
