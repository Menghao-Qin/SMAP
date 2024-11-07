# SMAP
Predict potential antimicrobial peptides from soil microbial.
First, you need to ensure that the protein sequence file you intend to predict has a sequence length of fewer than 200 amino acids. We recommend that you download and use SeqKit. You can achieve this by executing the command 'seqkit seq -M 200 input.fa > output.fa'.To perform predictions, utilize the script predict_amps.py, and ensure you have the necessary Python packages listed in requirements.txt. Begin by installing the required packages with the command 'pip install -r requirements.txt'. Next, download the model file amp_bilstm_classifier.pth and the script predict_amps.py to your local machine. You can then predict AMPs by entering the command 'python3 predict_amps.py data/input.fa model/amp_bilstm_classifier.pth local_camp.txt seq_camp.txt predicted_camp.fa'. Remember to replace data/input.fa and model/amp_bilstm_classifier.pth with the appropriate paths on your machine, and substitute input.fa with the filename of the sequence you wish to predict.
