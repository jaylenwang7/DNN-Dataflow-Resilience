from pathlib import Path
import helpers
import shutil
from os.path import exists

class MemInfo():
    def __init__(self, name: str):
        self.name = name
        self.mem_dict = dict(w=False, i=False, o=False)
        
    def set_dict(self, key: str):
        assert(key in ['w', 'i', 'o'])
        self.mem_dict[key] = True
        
    def get_dict(self):
        return self.mem_dict
    
    def get_name(self):
        return self.name

def parse_line(line: str):
    mem_obj = []
    spatial = False
    var_char = ''
    size = -1
    line_type = -1
    
    first_char = line[0]
    # loop line
    if first_char == '|':
        # get if is spatial or not
        if "Spatial" in line:
            spatial = True
        
        # get the var used
        var_ind = line.find("for") + 4
        var_char = line[var_ind].lower()
        
        # get the size of var
        size_ind = line.find(":")
        num_digs = 1
        while line[size_ind + num_digs].isdigit():
            num_digs += 1
        size = int(line[size_ind + 1:size_ind + num_digs])
        line_type = 0
    # under memory divider - don't need to do anything
    elif first_char == '-':
        pass
    # memory line
    else:
        line_words = line.split()
        mem_obj = MemInfo(line_words[0])
        # set whatever memory is contained to true in dict
        if "Weights" in line:
            mem_obj.set_dict('w')
            line_type = 1
        if "Inputs" in line:
            mem_obj.set_dict('i')
            line_type = 1
        if "Outputs" in line:
            mem_obj.set_dict('o')
            line_type = 1
    
    return line_type, mem_obj, spatial, var_char, size

def parse_map(file_name: str):
    map_file = open(file_name, 'r')
    lines = map_file.readlines()
    
    # loop through all lines and extract data
    file_data = []
    for i in range(len(lines)):
        line = lines[i]
        if line.strip():
            file_data.append(parse_line(line))
        
    loops = []
    divs = dict(w=[], i=[], o=[])
    name_dict = {}
    var_ind = 0 # keeps track of how many loop vars been processed
    # loop through all line data in file
    for line_data in file_data:
        # get the type of line it was
        line_type = line_data[0]
        
        # no information
        if line_type == -1:
            continue
        # loop var line
        elif line_type == 0:
            var = []
            # get the type and size of var
            var += [line_data[3], line_data[4]]
            # add spatial indicator if spatial
            if line_data[2]:
                var.append(True)
            var = tuple(var)
            
            # add to loops
            loops.append(var)
            var_ind += 1
        # mem info line
        elif line_type == 1:
            mem_obj = line_data[1]
            
            # add mem name to dict
            mem_name = mem_obj.get_name()
            name_dict[var_ind] = mem_name
            
            # add the mem divs to divs
            mem_dict = mem_obj.get_dict()
            for key, val in mem_dict.items():
                if val:
                    divs[key].append(var_ind)
        else:
            assert(False)
        
    return loops, divs, name_dict

def parse_files(dir: str, to_parse: str='**/*.map.txt', debug: bool=False):
    
    # get list of all txt files in dir
    pathlist = sorted(Path(dir).glob(to_parse), key=helpers.get_str_num)
    if not pathlist:
        assert(False and "check provided arch/model names are correct and directory structure")
    
    # get the layer number as specified by the file name
    layer_ids = [int(str(path).split('layer')[1].split('.')[0])-1 for path in pathlist]
    
    all_loops = {}
    all_divs = {}
    all_names = {}
    # loop through all files in dir provided
    for i in range(len(pathlist)):
        # get data for the file and add to lists
        f = pathlist[i]
        layer_id = layer_ids[i]
        loops, divs, names = parse_map(f)
        all_loops[layer_id] = loops
        all_divs[layer_id] = divs
        all_names[layer_id] = names
    
    if debug:
        for i in range(len(all_loops)):
            print("Layer " + str(i) + ":")
            print("Loops: " + str(all_loops[i]))
            print("Divs: " + str(all_divs[i]))
            print("Names: " + str(all_names[i]) + "\n")
        
    return all_loops, all_divs, all_names

def move_maps(arch_name: str, model_name: str, map_dir: str):

    overwrite = False

    dir = 'timeloop-injection/workspace/archs/' + arch_name + '/profiled_networks' + map_dir
    newdir = 'timeloop_maps/' + arch_name + '/' + model_name + '/'
    to_parse='**/*.map.txt'

    p = Path(newdir)
    p.mkdir(parents=True, exist_ok=True)

    pathlist = sorted(Path(dir).glob(to_parse), 
                      key=helpers.get_str_num)

    for f in pathlist:
        layer_num = str(f).split('layer')[1].split('/')[0]
        newfile = newdir + 'layer' + layer_num + '.map.txt'
        if not exists(newfile) or overwrite:
            print("Writing to: " + str(newfile))
            shutil.copy(f, newfile)
                