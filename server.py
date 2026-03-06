from openreward.environments import Server

from refseqtrain import RefSeqTrain

if __name__ == "__main__":
    server = Server([RefSeqTrain])
    server.run()
