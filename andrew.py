from fpgaconvnet.models.network import Network
import samo.cli
from fpgaconvnet.hls.generate.network import GenerateNetwork
import numpy as np
import PIL
import pickle
def main():
    # load network
    net = Network("comms", "models/andrew-sim-sim.onnx")

    # load the zedboard platform details
    net.update_platform("platforms/zedboard.json")

    # show latency, throughput and resource predictions
    print(f"predicted latency (us): {net.get_latency() * 1000000}")
    print(f"predicted throughput (img/s): {net.get_throughput()} (batch size={net.batch_size})")
    print(f"predicted resource usage: {net.partitions[0].get_resource_usage()}")

    # invoking the CLI from python
    samo.cli.main([
        "--model", "models/andrew-sim-sim.onnx",
        "--platform", "platforms/zedboard.json",
        "--output-path", "outputs/andrew_opt.json",
        "--backend", "fpgaconvnet",
        "--optimiser", "annealing",
	    "--enable_reconf", "false",
        "--objective", "latency"
    ])

    net.load_network("outputs/andrew_opt.json")  # TODO: change name
    net.update_partitions()

    # print the performance and resource predictions
    print(f"predicted latency (us): {net.get_latency() * 1000000}")
    print(f"predicted throughput (img/s): {net.get_throughput()} (batch size={net.batch_size})")
    print(f"predicted resource usage: {net.partitions[0].get_resource_usage()}")

    # create instance of the network
    gen_net = GenerateNetwork("andrew", "outputs/andrew_opt.json", "andrew-sim-sim.onnx")

    # generate hardware and create HLS project
    gen_net.create_partition_project(0)

    # load normalized test data (One set of 128 samples of Q and I)
    filehandler = open("CPFSKn.pkl", "rb")
    sample_input = pickle.load(filehandler).astype('f4')
    sample_input = np.stack([sample_input for _ in range(1)], axis=0 ) # duplicate across batch size
    sample_input = np.moveaxis(sample_input,-1, 1)

    # generate hardware
    gen_net.generate_partition_hardware(0)

    # run the partition's testbench
    gen_net.run_testbench(0, sample_input)

    # run co-simulation
    gen_net.run_cosimulation(0, sample_input)




if __name__ == '__main__':
    main()
