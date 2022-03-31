# class to handle a single loop variable type (so r, s, p, q)
class loop_var:
    # constructor
    def __init__(self, consts, orders, ltype, spatials, init_coord=0, stride=1):
        # initialize the current index of this var to 0 - this is which var it is handling
        # so 0 means x0 for X
        self.curr_ind = -1
        self.consts = consts            # list of consts for each var
        self.curr_num = 0               # where you are in the current var
        self.num_vars = len(consts)     # number of variables (taken from number of consts)
        self.orders = orders            # not actually being used
        self.spatials = spatials        # list of which vars are spatials (not currently being used)
        self.level_edge = -1            # 
        self.curr_coord = init_coord    # current actual coordinate (beginning of the tile being anaylyzed)
        self.init_coord = init_coord
        self.type = ltype               # this is what loop type this is (r, p, q, s, etc.)
        self.spatial = False            # marks if this loop is currently a spatial (not being used)
        self.stride = stride            # the stride for this var (default 1)
        
        # set up the offset
        curr_offset = 1
        off = []
        for i in range(len(consts) - 1, -1, -1):
            off.insert(0, curr_offset)
            curr_offset *= consts[i]
        self.off = off
        
        # set size - which will be the last offset (which is unused in off calculation)
        self.size = curr_offset
        self.working_set = range(0, 0)
        self.is_in = False
    
    # resets the current loop_var
    def reset(self):
        self.curr_ind = -1
        self.curr_coord = self.init_coord
        self.level_edge = -1
        self.working_set = range(0, 0)
        self.is_in = False

    def set_init_coord(self, init_coord):
        self.init_coord = init_coord
        self.curr_coord = init_coord
    
    # return the total number of elements in this dimension
    def get_size(self):
        return self.size
    
    # get the current offset for the curr_ind
    def get_off(self):
        # if you're above the top level loop_var
        if self.curr_ind == -1:
            return self.size
        
        return self.off[self.curr_ind]
    
    # is the given loop var a spatial loop
    def is_spatial(self):
        return self.spatials[self.curr_ind]
    
    # increment by the offset to get to the edge of the window
    # this returns the edge of what ONE ITERATION of this loop touches
    def get_edge(self):
        return self.curr_coord + self.get_off()
    
    # return the full range of values that will be iterated through
    # this returns the edge of what ALL ITERATIONS of this loop touches
    def get_full_edge(self):
        if self.curr_ind <= 0:
            off = self.size
        else:
            off = self.off[self.curr_ind - 1]
        return self.curr_coord + off
    
    # go to the next loop variable (down a for loop)
    def inc_lvar(self):
        self.curr_ind += 1
    
    # go to previous loop variable (up a for loop)
    def dec_lvar(self):
        self.curr_ind -= 1
    
    # increment to the next tile (next index) and return updated index
    def update_coord(self):
        self.curr_coord += self.get_off()
        # print(self.type + " = " + str(self.curr_coord))
        self.curr_num += 1
        return self.curr_coord
    
    # get the current coordinate
    def get_coord(self):
        return self.curr_coord*self.stride
    
    # get the constant of the current index
    def get_const(self):
        return self.consts[self.curr_ind]
    
    # call this before returning out of recursive loop
    def reset_lvar(self):
        # get the previous offset - which is what you want to get back to the last multiple of
        prev_off = 0
        if self.curr_ind == 0: # if you're the outer most loop, set to total size
            prev_off = self.size
        else: # else just get the last offset
            prev_off = self.off[self.curr_ind-1]
        
        # get to the last multiple of the last offset (go back up a loop level)
        self.curr_coord -= (self.curr_coord - 1) % prev_off + 1

        # go back up a level before returning
        self.curr_ind -= 1
    
    # check if the current loop_var's tile is in the range
    def is_in_bound(self, wind):
        # print("wind: " + str(wind))
        # print("curr_coord: " + str(self.curr_coord))
        
        # if self.curr_coord < wind[0]: # tile below range
        if self.get_edge() < wind[0]:
            return -1
        elif self.curr_coord > wind[-1]: # tile above range
            return 1
        else: # within range!
            return 0