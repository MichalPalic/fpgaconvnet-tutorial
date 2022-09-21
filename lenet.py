#from IPython.display import Image 
from fpgaconvnet.models.network import Network
import samo.cli
from fpgaconvnet.hls.generate.network import GenerateNetwork
import numpy as np
import PIL
import pickle
import PIL

def main():
    # load network
    net = Network("mnist", "models/mnist-12-sim.onnx")

    # load the zedboard platform details
    net.update_platform("platforms/zedboard.json")

    # show latency, throughput and resource predictions
    print(f"predicted latency (us): {net.get_latency() * 1000000}")
    print(f"predicted throughput (img/s): {net.get_throughput()} (batch size={net.batch_size})")
    print(f"predicted resource usage: {net.partitions[0].get_resource_usage()}")

    # invoking the CLI from python
    #samo.cli.main([
    #    "--model", "models/mnist-12-sim.onnx",
    #    "--platform", "platforms/zedboard.json",
    #    "--output-path", "outputs/mnist_opt.json",
    #    "--backend", "fpgaconvnet",
    #    "--optimiser", "rule",
    #    "--objective", "latency"
    #])

    net.load_network("outputs/mnist_opt.json")  # TODO: change name
    net.update_partitions()

    # print the performance and resource predictions
    print(f"predicted latency (us): {net.get_latency() * 1000000}")
    print(f"predicted throughput (img/s): {net.get_throughput()} (batch size={net.batch_size})")
    print(f"predicted resource usage: {net.partitions[0].get_resource_usage()}")

    # create instance of the network
    gen_net = GenerateNetwork("mnist", "outputs/mnist_opt.json", "models/mnist-12-sim.onnx")

    # generate hardware and create HLS project
    gen_net.create_partition_project(0)

    # show test image
    #im = Image('mnist_example.png')
    #display(im)

    # load test data
    input_image = PIL.Image.open("mnist_example.png") # load file
    input_image = np.array(input_image, dtype=np.float32) # convert to numpy
    input_image = np.expand_dims(input_image, axis=0) # add channel dimension
    #input_image = input_image/np.linalg.norm(input_image) # normalise
    input_image = np.stack([input_image for _ in range(1)], axis=0 ) # duplicate across batch size

    # generate hardware
    gen_net.generate_partition_hardware(0)

    # run the partition's testbench
    gen_net.run_testbench(0, input_image)

    # run co-simulation
    gen_net.run_cosimulation(0, input_image)




if __name__ == '__main__':
    main()
