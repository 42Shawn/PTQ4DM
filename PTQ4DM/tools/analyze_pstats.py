from ast import arg
import pstats
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('pstat_file',type=str)
args=parser.parse_args()

p=pstats.Stats(args.pstat_file)

# p.print_stats(10)

# p.sort_stats('calls').print_stats() #根据调用次数排序

# p.sort_stats('cumulative').print_stats()

p.sort_stats('time').print_stats(10)