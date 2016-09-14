import subprocess
import glob

from os import path
import argparse
import sys

  


def write_list_of_strings(list_of_strings, filename_with_path, add_newline = True):
    
    with open(filename_with_path, "w") as f:
        
        for one_string in list_of_strings:
            if add_newline:
                f.write(one_string + "\n")
            else:
                f.write(one_string)
                

def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ptb_patched_dir', help="Directory containing patched wsj")
    parser.add_argument('--penn_converter_dir', help="Directory containing pennconverter.jar")
    parser.add_argument('--data_output_dir', help="Data directory for the dependency trees")
    parser.add_argument('--data_output_logs_dir', help="Data directory for logs created by pennconverter.jar")
    
    args = parser.parse_args(arguments)
    
    ptb_patched_dir = args.ptb_patched_dir 
    penn_converter_dir = args.penn_converter_dir
    data_output_dir = args.data_output_dir 
    data_output_logs_dir = args.data_output_logs_dir 
    

    all_dependency_trees_by_section = []
    
    path_list = glob.glob(path.join(ptb_patched_dir, "[0-9][0-9]/"))
    
    for path_for_one_section in path_list:
        section_name = path_for_one_section[-3:-1]
        assert section_name.isdigit()
        print "Currently processing section %s" % section_name
        
        one_section_dependency_tree_path_filename = path.join(data_output_dir, "wsj_%s_dep.txt" % section_name)
        
        dependency_trees_separated_by_newline = []
        
        
        
        file_list = glob.glob(path_for_one_section+"*.mrg")
        
        for one_file in file_list:
            filename = one_file[one_file.rfind("/")+1:]
            log_path_filename = path.join(data_output_logs_dir, "log_%s_%s.txt" % (section_name, filename))
            
            sub_call = subprocess.check_output("java -jar %s/pennconverter.jar -log " % (penn_converter_dir) + log_path_filename + "  -verbosity 1 -stopOnError=true -format=conllx < " + one_file , shell=True)
    
            
            dependency_trees_separated_by_newline.append(sub_call.strip())
            dependency_trees_separated_by_newline.append("\n")
            
        write_list_of_strings(dependency_trees_separated_by_newline, one_section_dependency_tree_path_filename, add_newline = True)
        all_dependency_trees_by_section.append(dependency_trees_separated_by_newline)
            
        

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
