from loop_var import LoopVar

# represents whole loop nest for a certain architecture/mapping
class Loop():
    # definitions for things that don't change from different class instances
    types = ['q', 'p', 's', 'r', 'c', 'm']
    input_types = ['q', 'p', 's', 'r']
    output_dict = {'m':0, 'q':1, 'p':2, 's':1, 'r':2}
    weight_types = ['r', 's']
    output_types = ['q', 'p']
    working_types = ['q', 'p', 's', 'r', 'm']
    d_types = ['i', 'w', 'o']
    
    # constructor
    def __init__(self, input_vars_in, mem_dividers, d_type='i', input_strides=[1, 1], sizes=[], paddings=[0,0], out_file='', serial=False):
        assert(d_type in self.d_types)
        self.input_vars_in = input_vars_in
        self.spatial_m_val = 0
        self.out_file = out_file
        loop_consts = {k: [] for k in self.types}
        og_sizes = {k: 1 for k in self.types}
        loop_orders = {k: [] for k in self.types}
        spatials    = {k: [] for k in self.types}
        has_spatial = {k: False for k in self.types}
        num_spatial = {k: 1 for k in self.types}
        self.has_spatial = has_spatial
        self.num_spatial = num_spatial
        self.og_sizes = og_sizes
        self.get_og_sizes(input_vars_in)
        self.d_type = d_type
        assert(len(input_strides) == 2)
        assert(len(paddings) == 2)
        self.strides = input_strides

        self.WITH_SPATIAL = False
        self.SERIAL = serial
        self.final_spatial = 0
        self.final_spatials = []
        self.spatial_consts = []
        self.is_spatial = []
        self.num_macs = 0

        # create a list for each of the variables where each list element points to a loop_var object
        # these objects are shared between different levels of the same var (x0, x1, x2, ...)
        input_vars = 0
        # also process with spatials for output injection
        # if not self.WITH_SPATIAL and not d_type == 'o' and not self.SERIAL:
        input_vars = self.preprocess_input_vars(input_vars_in)
        # run this if output injection - since can just run the spatials sequentially
        # else:
        #     input_vars = self.preprocess_input_vars_w_spatial(input_vars_in)
        
        self.mem_inds = []
        for mem_div in mem_dividers:
            self.mem_inds.append((mem_div, self.get_xformed_mem_ind(mem_div)))

        # for i in range(len(input_vars_in)):
        #     lvar = input_vars_in[i]
        #     if len(lvar) >= 3:
        #         has_spatial[lvar[0]] = True
        
        for i in range(len(input_vars)):
            lvar = input_vars[i]
            assert(lvar[0] in self.types)
            tname = lvar[0]
            loop_consts[tname].append(lvar[1])
            loop_orders[tname].append(i)
            if len(lvar) < 3:
                spatials[tname].append(False)
            else:
                spatials[tname].append(True)

        spatial_info = []
        # loop through each spatial ind
        for i in range(len(self.spatial_inds)):
            # get unxformed ind
            sp_ind = self.spatial_inds[i][1]
            # get spatial ind
            sp_var = input_vars_in[sp_ind]
            # get type of spatial
            sp_type = sp_var[0]
            # get spatial const
            sp_const = sp_var[1]
            # keep track of total size (including spatials) of this type
            # and the increment
            sp_size  = 1
            sp_inc   = 1
            sp_steps = 1
            
            # loop through all lvars
            for j in range(len(input_vars_in)):
                lvar = input_vars_in[j]
                # if same type
                if lvar[0] == sp_type:
                    curr_const = lvar[1]
                    # update total size of this var
                    sp_size *= curr_const
                    # if inside the current spatial, update inc
                    if j > sp_ind:
                        sp_inc *= curr_const
                    # number of times the spatial is called from above
                    elif j < sp_ind:
                        sp_steps *= curr_const
            # what you put inside is:
            # (total size of spatial var, size of everything inside, const value)
            spatial_info.append((sp_size, sp_inc, sp_steps, sp_const))
        self.spatial_info = spatial_info

        # print("spatial_inds: " + str(self.spatial_inds))
        # print("spatial_info: " + str(spatial_info))     
        # print("loop_consts: " + str(loop_consts))
        # print("spatials: " + str(spatials))
        # print("loop_orders: " + str(loop_orders))
        # print("mem_inds: " + str(self.mem_inds))
        # print("strides: " + str(self.strides))
        self.loop_consts = loop_consts
        self.spatials = spatials
        self.loop_orders = loop_orders
        
        if self.out_file:
            with open(self.out_file, 'a') as f:
                f.write("spatial_inds: " + str(self.spatial_inds) + "\n")
                f.write("spatial_info: " + str(spatial_info) + "\n") 
                f.write("loop_consts: " + str(loop_consts) + "\n")
                f.write("spatials: " + str(spatials) + "\n")
                f.write("loop_orders: " + str(loop_orders) + "\n")
                f.write("mem_inds: " + str(self.mem_inds) + "\n")
                f.write("strides: " + str(self.strides) + "\n")
                f.write("d_type: " + d_type + "\n")
        
        loop_classes = {}
        # set the strides for p and q
        for ltype in self.types:
            stride = 1
            if ltype == 'p':
                stride = self.strides[1]
            elif ltype == 'q':
                stride = self.strides[0]

            loop_classes[ltype] = LoopVar(loop_consts[ltype], loop_orders[ltype], ltype, spatials[ltype], stride=stride)
        loop_vars = {}
        for i in range(len(input_vars)):
            loop_vars[i] = loop_classes[input_vars[i][0]]
        self.loop_classes = loop_classes
        self.loop_vars = loop_vars
        
        # current loop_var being looked at
        self.curr_var_ind = -1
        self.curr_var = loop_vars[0]
        self.num_vars = len(self.loop_vars)
        
        # initialize sizes based on the loop nest vars
        self.m_size = loop_classes['m'].get_size()
        self.c_size = loop_classes['c'].get_size()
        self.s_size = loop_classes['s'].get_size()
        self.r_size = loop_classes['r'].get_size()
        self.q_size = loop_classes['q'].get_size()
        self.p_size = loop_classes['p'].get_size()
        self.h_size = (self.q_size-1)*self.strides[0] + self.s_size
        self.w_size = (self.p_size-1)*self.strides[1] + self.r_size
        
        # sizes of the data
        self.weight_size = (self.m_size, self.s_size, self.r_size)
        self.out_size = (self.m_size, self.q_size, self.p_size)
        self.input_size = (self.c_size, self.h_size, self.w_size)
        
        # get original sizes - so ignoring any spatials or removal of loop layers (pre-op)
        self.paddings = [0, 0]
        if not sizes:
            og_h_size = (self.og_sizes['q']-1)*self.strides[0] + self.og_sizes['s']
            og_w_size = (self.og_sizes['p']-1)*self.strides[1] + self.og_sizes['r']
            self.original_sizes = [self.og_sizes['m'], self.og_sizes['c'], self.og_sizes['s'], self.og_sizes['r'], 
                                   self.og_sizes['q'], self.og_sizes['p'], og_h_size, og_w_size]
            
            print("Original size is " + str(self.original_sizes))
        else:
            # have to add on padding when not inferring directly from output size
            self.original_sizes = sizes
            self.paddings = paddings
        
        # set of indices in the full output window
        self.out_set = []
        # dividers for this run
        self.out_dividers = []
        # all dividers for all mem indices
        self.all_dividers = []
        # all sites for all of the dividers for all mem indices
        self.all_out_sites = []
        self.all_timed_sites = []
        
        # used to keep track of index in the out_set - for divider
        self.curr_divider = 0
        self.added = True
        
        self.window = 0
        self.is_in = True
        self.first = True
        
        # self.hset = 0
        # self.vset = 0
        self.hset = 0
        self.vset = 0
        self.mset = 0

        self.hsets = []
        self.vsets = []
        self.msets = []
        self.mac_ins = []
        self.mac_curr_set = []
        self.mac_out_sets = []
        if self.WITH_SPATIAL:
            self.init_sets()
        # says which mac currently targeting
        self.mac_ind = 0
        self.spatial_ind = -1
        
    def __str__(self):
        out_str = ""
        out_str += "input_vars: " + str(self.input_vars_in) + "\n"
        out_str += "spatial_inds: " + str(self.spatial_inds) + "\n"
        out_str += "spatial_info: " + str(self.spatial_info) + "\n"
        out_str += "loop_consts: " + str(self.loop_consts) + "\n"
        out_str += "spatials: " + str(self.spatials) + "\n"
        out_str += "loop_orders: " + str(self.loop_orders) + "\n"
        out_str += "mem_inds: " + str(self.mem_inds) + "\n"
        out_str += "strides: " + str(self.strides) + "\n"
        out_str += "d_type: " + self.d_type + "\n"
        return out_str
    
    # reset this loop - called between injections
    def reset(self):
        self.curr_var_ind = -1
        self.window = 0
        self.is_in = True
        self.curr_out_set = []
        self.curr_var = self.loop_vars[0]
        self.hset = 0
        self.vset = 0
        self.mset = 0
        self.spatial_ind = -1
        self.mac_ind = 0
        if self.WITH_SPATIAL:
            self.init_sets()
        self.inj_level = 0
        self.curr_divider = 0
        self.out_dividers = []
        for _, lclass in self.loop_classes.items():
            lclass.reset()
    
    # perform a hard reset - this is for when the injection location changes
    def hard_reset(self):
        self.reset()
        self.original_window = []
        self.first = True
        self.all_dividers = []
        self.out_set = []
        self.all_out_sites = []
        self.all_timed_sites = []
    
    # get the original sizes of each variable
    def get_og_sizes(self, input_vars):
        for i in range(len(input_vars)):
            lvar = input_vars[i]
            ltype = lvar[0]
            self.og_sizes[ltype] *= lvar[1]
    
    # process the given loop to remove loops not required 
    def preprocess_input_vars(self, input_vars):
        new_vars = []
        taken_inds = []
        spatial_inds = []
        xformed_ind = 0
        for i in range(len(input_vars)):
            lvar = input_vars[i]
            if lvar[0] not in self.working_types:
                continue

            # don't process spatials (which is the len(lvar) check)
            if len(lvar) < 3:
                new_vars.append(lvar)
                taken_inds.append(i)
                xformed_ind += 1
            # add to spatial ind array if spatial - also add output index
            else:
                spatial_inds.append((xformed_ind, i, self.output_dict[lvar[0]], lvar[0]))
                self.has_spatial[lvar[0]] = True
                self.num_spatial[lvar[0]] *= lvar[1]

        self.taken_inds = taken_inds
        self.spatial_inds = spatial_inds
        return new_vars

    def preprocess_input_vars_w_spatial(self, input_vars):
        new_vars = []
        taken_inds = []
        spatial_inds = []
        xformed_ind = 0
        final_spatial = 0
        spatial_num = 1

        working_set = []
        if self.d_type == 'o':
            working_set = self.types
        else:
            working_set = self.working_types

        # total_spatials = 1
        for i in range(len(input_vars)):
            lvar = input_vars[i]
            
            if lvar[0] in working_set:
                new_vars.append(input_vars[i])
                taken_inds.append(i)
                xformed_ind += 1

                # add to spatial ind array if spatial - also add output index
                if len(lvar) >= 3:
                    self.is_spatial.append(True)
                    final_spatial = xformed_ind
                    spatial_num *= lvar[1]

                    spatial_inds.append((xformed_ind, i, self.output_dict[lvar[0]]))
                else:
                    self.is_spatial.append(False)
                self.spatial_consts.append(spatial_num)

        self.is_spatial.append(False)
        self.num_macs = spatial_num
        print("num_macs: " + str(self.num_macs))
        for i in range(len(self.spatial_consts)):
            self.spatial_consts[i] = spatial_num // self.spatial_consts[i]

        self.final_spatial = final_spatial
        self.taken_inds = taken_inds
        self.spatial_inds = spatial_inds
        return new_vars
    
    # transforms the given injection level into the level after processing
    def get_xformed_mem_ind(self, ind):
        for i in range(len(self.taken_inds)):
            if ind <= self.taken_inds[i]:
                return i
        return len(self.taken_inds)
    
    # index is the index of the injection location
    # GET THE WINDOW AT THE OUTPUT OF THE 'd_type' MEMORY AT 'index'
    def set_window(self, index):
        print("Setting window...")
        # if weight
        if self.d_type == 'w':
            # return ranges
            # for m it's just that single m
            # for s and r it's just the size of the output (so basically an entire output channel)
            self.window = (range(index[0], index[0]+1), range(self.out_size[1]), range(self.out_size[2]))
            return
        # if input
        elif self.d_type == 'i':
            # for a chosen dimension (H or W):
            # W = weight kernel size
            # S = stride
            # I = target index value
            # L = input size
            # returns 
            def get_input_window(W, S, I, L, offset=0):
                # i is the beginning of the input window
                # so [i, i+W) gives the current window
                i = offset
                # current output index being calculated
                o = 0
                # 
                out_off = 0
                # range of output indices that use the given target index
                ranges = []
                # whether window is covering the target index
                in_range = False
                while True:
                    if i + W > L and len(ranges) == 2:
                        ranges.append(o)
                        break
                    # loop until you get in range
                    if I in range(i, i+W):
                        # if first time in range - append beginning of window
                        if not in_range:
                            in_range = True
                            ranges.append(o)
                            out_off = I - i
                    # if out of range
                    else:
                        # if you were in range - append end of window
                        if in_range:
                            ranges.append(o)
                            break
                    o += 1
                    i += S

                    assert(i < 1000)

                if len(ranges) == 1:
                    ranges.append(o)
                    
                out_range = 0
                try:
                    out_range = range(ranges[0], ranges[1])
                except:
                    print(ranges)
                    print("W: " + str(W) + ", S: " + str(S) + ", I: " + str(I) + ", L: " + str(L) + ", off: " + str(offset))
                    assert(False)
                return out_range, out_off
            
            y_off = x_off = 0
            s_sp = False
            r_sp = False

            if not self.WITH_SPATIAL and not self.SERIAL:
                if self.has_spatial['s']:
                    s_sp = True
                if self.has_spatial['r']:
                    r_sp = True
            
            y_range, y_off = get_input_window(self.og_sizes['s'], self.strides[0], index[1], self.input_size[1], offset=y_off)
            x_range, x_off = get_input_window(self.og_sizes['r'], self.strides[1], index[2], self.input_size[2], offset=x_off)
            
            if s_sp:
                self.loop_classes['s'].set_init_coord(y_off)
            if r_sp:
                self.loop_classes['r'].set_init_coord(x_off)
            
            self.window = (range(self.out_size[0]), y_range, x_range)
            return
        elif self.d_type == 'o':
            self.window = (range(index[0], index[0]+1), range(index[1], index[1]+1), range(index[2], index[2]+1))
        else:
            assert(False and "not supported window type")
    
    # checks if the tile that's been iterated to is in bounds (specified by ranges)
    # must be called after 'inject()' has been called
    # this really just checks if the simulation is in the bounds of the output index
    # so it only checks with q and p
    def is_in_bound(self):
        t = self.get_type()
        
        # only check q and p since you only really care about the output indices
        if t == 'q':
            wind = self.window[1]
        elif t == 'p':
            wind = self.window[2]
        else:
            return 0

        return self.curr_var.is_in_bound(wind)

    def init_sets(self):
        for i in range(self.num_macs):
            self.hsets.append(range(0,0))
            self.vsets.append(range(0,0))
            self.msets.append(range(0,0))
            self.mac_ins.append(True)
            self.mac_curr_set.append([])
            self.mac_out_sets.append([])
    
    # update the horizontal bounds
    def update_hset(self):
        t = self.get_type()
        add_on = [0, 0]

        # if weight injection, just use r and s
        if self.d_type == 'w':
            self.hset = range(self.curr_var.get_coord(), self.curr_var.get_full_edge())
            self.vset = range(self.loop_classes['s'].get_coord(), self.loop_classes['s'].get_edge())
            return
        elif self.d_type == 'i':
            if t == 'p':
                add_on[0] = self.loop_classes['r'].get_coord()
                add_on[1] = self.loop_classes['r'].get_edge()
            elif t == 'r':
                add_on[0] = self.loop_classes['p'].get_coord()
                add_on[1] = self.loop_classes['p'].get_edge()
            
            # use the full range for the current var - since the current var can iterate through its whole bounds
            self.hset = range(self.curr_var.get_coord()+add_on[0],
                              self.curr_var.get_full_edge()+add_on[1]-1)
            self.vset = range(self.loop_classes['s'].get_coord()+self.loop_classes['q'].get_coord(),
                              self.loop_classes['s'].get_edge()+self.loop_classes['q'].get_edge()-1)
            self.mset = range(self.loop_classes['m'].get_coord(), self.loop_classes['m'].get_edge())
            return
        elif self.d_type == 'o':
            pass
        else:
            assert(False)
    
    # update the vertical bounds
    def update_vset(self):
        t = self.get_type()
        add_on = [0, 0]

        # if weight injection, just use r and s
        if self.d_type == 'w':
            self.hset = range(self.curr_var.get_coord(), self.curr_var.get_full_edge())
            self.vset = range(self.loop_classes['r'].get_coord(), self.loop_classes['r'].get_edge())
            return
        elif self.d_type == 'i':
            if t == 'q':
                add_on[0] = self.loop_classes['s'].get_coord()
                add_on[1] = self.loop_classes['s'].get_edge()
            elif t == 's':
                add_on[0] = self.loop_classes['q'].get_coord()
                add_on[1] = self.loop_classes['q'].get_edge()
            
            # use the full range for the current var - since the current var can iterate through its whole bounds
            self.vset = range(self.curr_var.get_coord()+add_on[0],
                              self.curr_var.get_full_edge()+add_on[1]-1)
            self.hset = range(self.loop_classes['r'].get_coord()+self.loop_classes['p'].get_coord(),
                              self.loop_classes['r'].get_edge()+self.loop_classes['p'].get_edge()-1)
            self.mset = range(self.loop_classes['m'].get_coord(), self.loop_classes['m'].get_edge())
            return
        elif self.d_type == 'o':
            pass
        else:
            assert(False)
        
    def update_mset(self):
        self.vset = range(self.loop_classes['s'].get_coord()+self.loop_classes['q'].get_coord(),
                          self.loop_classes['s'].get_edge()+self.loop_classes['q'].get_edge()-1)
        self.hset = range(self.loop_classes['r'].get_coord()+self.loop_classes['p'].get_coord(),
                          self.loop_classes['r'].get_edge()+self.loop_classes['p'].get_edge()-1)
        self.mset = range(self.curr_var.get_coord(),
                          self.curr_var.get_full_edge())

    def check_mac_ind(self):
        assert(self.mac_ind <= self.num_macs)
        assert(self.mac_ind >= 0)

    def get_spatial_const(self):
        return self.spatial_consts[self.curr_var_ind]

    def get_mac_range(self):
        self.check_mac_ind()
        start = self.mac_ind
        tot_range = self.get_spatial_const()
        assert(start + tot_range < self.num_macs)
        mac_range = range(start, start+tot_range)
        return mac_range

    def set_is_in(self, is_in):
        if not self.WITH_SPATIAL:
            self.is_in = is_in
            return
        
        mac_range = self.get_mac_range()
        for i in mac_range:
            self.mac_ins[i] = is_in
    
    # update the working sets
    def update_set(self):
        t = self.get_type()
        if (t == 'q' and self.d_type=='i') or t == 's':
            self.update_vset()
        elif (t == 'p' and self.d_type=='i') or t == 'r':
            self.update_hset()
        else:
            self.update_mset()
        
        if self.WITH_SPATIAL:
            self.check_mac_ind()
            self.vsets[self.mac_ind] = self.vset
            self.hsets[self.mac_ind] = self.hset
            self.msets[self.mac_ind] = self.mset
    
    # checks whether the iterated to loop keeps the inected injection index in the working set
    def check_in_set(self):
        if not self.WITH_SPATIAL:
            in_m = self.inj_ind[0] in self.mset
            in_v = self.inj_ind[1] in self.vset
            in_h = self.inj_ind[2] in self.hset
            return in_m and in_v and in_h
            # return in_v and in_h
        else:
            self.check_mac_ind()
            in_m = self.inj_ind[0] in self.msets[self.mac_ind]
            in_v = self.inj_ind[1] in self.vsets[self.mac_ind]
            in_h = self.inj_ind[2] in self.hsets[self.mac_ind]
            return in_m and in_v and in_h
    
    # returns the type of the current loop variable (as a character)
    def get_type(self):
        return self.curr_var.type
    
    # function adds the current error set to the out_sets and resets the curr_out_set to empty
    def add_new_set(self):
        # any time a variable below the injection index changes - you just change the out_set that you're using
        # append the old one
        # append a new error set
        if self.WITH_SPATIAL:
            mac_range = self.get_mac_range()
            for i in mac_range:
                if self.mac_curr_set[i]:
                    self.mac_out_sets[i].append(self.mac_curr_set[i])
                    self.mac_curr_set[i] = []
            return

        if not self.added:
            self.out_dividers.append(self.curr_divider)
            self.added = True

    def is_below_memory_level(self):
        return self.curr_var_ind > self.inj_level

    def check_is_in(self):
        if not self.WITH_SPATIAL:
            return self.is_in
        else:
            return self.mac_ins[self.mac_ind]
    
    # this function handles adding new error sets if needed
    # when is it needed:
    #   * If you're at or above the injection level and you bring in the injected value (wasn't already
    #     in the working set)
    def add_if_set(self):

        # if below the memory injection level then don't need to do anything
        if self.is_below_memory_level():
            return

        # update the working set (at level because you're at the memory interface)
        self.update_set()
        # check whether the update results in the index being in or out of bounds
        is_in_bound = self.check_in_set()
        
        # if you've gone out of bounds - add a new set
        # if not is_in_bound and not is_spatial:
        if not is_in_bound:
            self.set_is_in(False)
            self.add_new_set()
        # if in bound or spatial
        else:
            # check if not in before - add new set
            if not self.check_is_in():
                self.add_new_set()
            self.set_is_in(True)

    # is current var a spatial
    def is_curr_spatial(self):
        return self.is_spatial[self.curr_var_ind]

    # is the next var down a spatial (to be called next)
    def is_next_spatial(self):
        if self.curr_var_ind >= len(self.is_spatial) - 1:
            return False
        return self.is_spatial[self.curr_var_ind + 1]

    # are you at the final spatial level
    def is_final_spatial(self):
        is_final = self.curr_var_ind == self.final_spatial
        if is_final:
            assert(self.is_curr_spatial())
        return is_final
    
    def update_mac_ind(self):
        if not self.WITH_SPATIAL:
            return

        # only concerned with spatials at or above memory level
        if self.is_below_memory_level():
            return

        print("curr_ind = " + str(self.curr_var_ind))
        if self.is_curr_spatial():
            # print(self.get_curr_const())
            self.mac_ind += self.get_curr_const()
        else:
            if self.is_next_spatial():
                self.mac_ind -= self.get_spatial_const()
        print("mac_ind = " + str(self.mac_ind) + "\n")
    
    # based on the loop that you're currently in - update the index
    # returns True if you're OOB - False otherwise
    def update_coord(self):
        # update the loop_vars coord
        self.curr_var.update_coord()

        # check if everything is in_bound
        in_bound = self.is_in_bound()

        # update the mac while you update the coord
        if self.WITH_SPATIAL:
            self.update_mac_ind() 

        if in_bound > 0: # if OOB of window (after range)
            # if you're outside the memory level, and you've gone out
            # of bounds, then the value has been evicted
            if self.curr_var_ind < self.inj_level:
                self.is_in = False
            return True
                
        return False
    
    # go to the next loop_var (increment it) and also set the curr var to that loop_var instance
    # call this before going to the next inner loop
    def inc_curr_var(self):
        # increment the index
        self.curr_var_ind += 1
        # set the new curr_var to this new loop_var
        self.curr_var = self.loop_vars[self.curr_var_ind]
        # increment this loop_var's var
        self.curr_var.inc_lvar()

    
    
    # go back to previous loop
    # call this when returning from loop calls - after for loop
    def dec_curr_var(self):
        # if you're back at DRAM - add new set
        if self.curr_var_ind == 0:
            self.add_new_set()
            return

        # within the loop_var that is being exited, go to one outer level and reset the current one to 0
        self.curr_var.reset_lvar()
        # go to an outer loop_var
        self.curr_var = self.loop_vars[self.curr_var_ind-1]
        # decrement the index
        self.curr_var_ind -= 1
    
    # get the constant of the current loop_var
    def get_curr_const(self):
        return self.curr_var.get_const()

    # get the stride
    def get_stride(self):
        return self.strides
    
    # functions for getting the current input/weight/output 
    # coordinates by looking into respective classes
    def get_input_coord(self):
        # take into account stride in calculation
        # if stride is [s_y, s_x], then:
        # [H, W] = [q*s_y + s, p*s_x + r]
        # this is done implicityly within the get_coord functions of p and q
        return (self.loop_classes['q'].get_coord() + self.loop_classes['s'].get_coord(), 
                self.loop_classes['p'].get_coord() + self.loop_classes['r'].get_coord())
    def get_weight_coord(self):
        return (self.loop_classes['m'].get_coord(),
                self.loop_classes['s'].get_coord(), 
                self.loop_classes['r'].get_coord())
    def get_full_weight_coord(self):
        return (self.loop_classes['m'].get_coord(),
                self.loop_classes['c'].get_coord(),
                self.loop_classes['s'].get_coord(), 
                self.loop_classes['r'].get_coord())
    def get_output_coord(self):
        m_val = self.loop_classes['m'].get_coord()

        if not self.WITH_SPATIAL and not self.SERIAL:
            if self.d_type == 'w' and self.has_spatial['m']:
                m_val = self.spatial_m_val

        return (m_val, 
                self.loop_classes['q'].get_coord()//self.strides[0], 
                self.loop_classes['p'].get_coord()//self.strides[1])
    
    # compare if the current index is the injected one
    # must be called after 'inject_input'
    def compare_inj(self):
        # get coord based on datatype
        if self.d_type == 'i':
            return self.get_input_coord() == (self.inj_ind[1], self.inj_ind[2])
        elif self.d_type == 'w':
            return self.get_weight_coord() == self.inj_ind
        else: # d_type is 'o'
            return self.get_output_coord() == self.inj_ind

    # transform to given user injection index into what can be used internally
    def transform_inj_ind(self, inj_ind):
        # copy to list to change values
        new_ind = list(inj_ind)
        if not self.WITH_SPATIAL and not self.SERIAL:
            # if weight injection
            if self.d_type == 'w':
                assert(len(inj_ind) == 4)
                # can get rid of the c
                # get m and set rest to 0
                new_ind = [new_ind[0], 0, 0]
                
                # if m is spatial, then set m to 0 too and same the og value
                if self.has_spatial['m']:
                    # save the m value - since this shows up in output
                    self.spatial_m_val = inj_ind[0]
                    new_ind[0] = 0
            elif self.d_type == 'i':
                assert(len(inj_ind) == 3)
                new_ind[1] //= self.num_spatial['q']
                new_ind[2] //= self.num_spatial['p']
            elif self.d_type == 'o':
                assert(len(inj_ind) == 3)
                new_ind = inj_ind
            else:
                assert(False)
        else:
            if self.d_type == 'w':
                # can get rid of the c
                # get m and set rest to 0
                new_ind = [new_ind[0], new_ind[2], new_ind[3]]
            elif self.d_type == 'i':
                new_ind = new_ind
        print("Transformed index = " + str(tuple(new_ind)))
        return tuple(new_ind)
    
    # get the ranges of the output window but with the original, unaltered (with spatial) sizes
    def get_original_window(self, inj_ind):
        return get_window(inj_ind, self.original_sizes, self.strides, self.paddings, self.d_type)
    
    # run an input injection at the injection input 'inj_ind' represented by (C, H, W) at the level before 'inj_level'
    # inj_level is an index (so the first, second, third, etc. inj_level)
    def inject(self, inj_ind, inj_level):
        if self.out_file:
            with open(self.out_file, 'a') as f:
                f.write("inj_ind: " + str(inj_ind) + "\n")
                
        # reset before performing injection
        self.reset()
        
        # for later processing, get the original window
        self.original_window = self.get_original_window(inj_ind)

        # transform the inj_ind (to handle spatials, strides, etc.)
        inj_ind = self.transform_inj_ind(inj_ind)
        
        # set new class variables to hold injection info
        self.inj_ind = inj_ind
        self.inj_level = self.mem_inds[inj_level][1]
        
        # calculate the window to use in the loop
        self.set_window(inj_ind)
        
        # run injection
        self.run_loop()
        
        self.first = False
        
        # return all the output sites
        return self.out_set, self.out_dividers
    
    # perform an injection on all the mem_inds
    # record all the dividers and sites for each divider
    def inject_full(self, inj_ind):
        print("Injecting full... at " + str(inj_ind))
        if self.out_file:
            with open(self.out_file, 'a') as f:
                f.write("inj_ind: " + str(inj_ind) + "\n")
        
        # do a hard reset since starting a whole new injection
        self.hard_reset()
        
        # for later processing, get the original window
        self.original_window = self.get_original_window(inj_ind)
        print("Original window is " + str(self.original_window) + "...")

        # transform the inj_ind (deal with spatials, etc.)
        inj_ind = self.transform_inj_ind(inj_ind)

        for i in range(len(self.mem_inds)):
            # reset before performing injection
            self.reset()

            # set inj_ind
            self.inj_ind = inj_ind

            # calculate the window to use in the loop
            self.set_window(inj_ind)

            # set new class variables to hold injection info
            self.inj_level = self.mem_inds[i][1]

            # run injection
            self.run_loop()
            
            # record all of the divider sets
            self.all_dividers.append(self.out_dividers)
            # record all of the out sites for the dividers
            self.all_out_sites.append(self.get_sites())
            
            # don't need to record out_set again
            self.first = False
            
        return self.all_dividers, self.all_out_sites
    
    # must be called after inject_input_full and get_all_sites
    # this compiles them for a single mem_ind - marked by mem_ind (mem_ind 
    # will index into all_dividers to get a set of dividers)
    def get_timed_sites(self, mem_ind):
        # if not all dividers/sites generated yet - generate them
        if not self.all_out_sites or not self.all_dividers:
            assert(False)
        
        # get the dividers for the inquired div
        divs = self.all_dividers[mem_ind]
        # get the divided sets for this mem_ind
        div_sets = self.all_out_sites[mem_ind]
        
        # if last level - just return the sites for this one
        if mem_ind == len(self.all_dividers)-1:
            # NEED TO CHANGE THIS TO ADD AN EXTRA LIST DIMENSION
            unsqueezed_sites = []
            for site in self.all_out_sites[mem_ind]:
                unsqueezed_sites.append([site])
            return unsqueezed_sites
        
        # get the next set of dividers
        next_divs = self.all_dividers[mem_ind+1]
        
        # loop backwards through the sites and collect along the dividers of the previous
        
        # a list of lists of lists of timed error sites (list of cur_sites)
        ret_sites = []
        # index starting from the last of the next divisions
        n = len(next_divs)-2
        next_div = 0
        try:
            next_div = next_divs[len(next_divs)-1]
        except:
            print("next_divs: " + str(next_divs))
            print("all dividers: " + str(self.all_dividers))
            assert(False)
        # loop through each of the divided sets (div_set) backwards
        i = len(divs)-2
        while i >= -1:
            # get the current div (start of the set)
            if i == -1:
                curr_div = 0
            else:
                curr_div = divs[i]
            
            # get the end of the set
            end_div = divs[i+1]
            
            # accumulate the timed set here
            div_time_sites = []
            
            # loop backwards through next_divs until it goes past curr_div
            while True:
                # get next_div
                if n < -1:
                    break
                elif n == -1:
                    next_div = 0
                else:
                    next_div = next_divs[n]
                    
                if next_div < curr_div:
                    break
                    
                # append the slice from next_div to end_div
                div_time_sites.insert(0, self.out_set[next_div:end_div])
                # go to next next_div
                n -= 1
            
            # add to total return sites
            ret_sites.insert(0, div_time_sites)
            # got to next curr_div
            i -= 1
        
        return ret_sites
    
    def get_all_timed_sites(self):
        all_timed_sites = []
        for i in range(len(self.mem_inds)):
            ret_sites = self.get_timed_sites(i)
            all_timed_sites.append(ret_sites)
        
        return all_timed_sites
            
    # given a set of dividers, this returns the individual sets
    # sets the self.out_sites
    def get_sites(self):
        # current index into the out_set
        cur_ind = 0
        # list to collect sites
        out_sites = []
        # loop through the list of dividers
        for div in self.out_dividers:
            # collect the sites within the divider
            curr_sites = []
            # loop until you hit the divider
            while not cur_ind == div:
                # append the current index to this curr_sites
                ind = self.out_set[cur_ind]
                curr_sites.append(ind)
                cur_ind += 1
            # add curr_sites
            out_sites.append(curr_sites)
        
        # set class's out_sites and return
        return out_sites
                
    
    # recursive loop function for getting input sites
    def run_loop(self):
        # go to next inner loop
        self.inc_curr_var()
        # based on the current loop - see if a new set needs to be added
        self.add_if_set()
        
        # if in the innermost level - this is the recursive base case
        if self.curr_var_ind == self.num_vars - 1:
            # see if the current input index matches the injection
            for i in range(self.get_curr_const()):
                # compare the current index to the injection index
                if self.compare_inj():
                    # if it's a match - inc divider
                    self.curr_divider += 1
                    # mark this new divider as not being added yet
                    self.added = False
                    # if this is first injection being performed - add to output set
                    if self.first:
                        if self.d_type in ['i', 'w']:
                            self.out_set.append(self.get_output_coord())
                        # if output injection - then you need the weight sites
                        else:
                            self.out_set.append(self.get_full_weight_coord())

                # update coord and check if OOB
                if self.update_coord():
                    break
            
            # go up a level before returning
            self.dec_curr_var()
                
            # return out
            return False
        
        # recursively call up to the const number of times (call the inner loop_var)
        for i in range(self.get_curr_const()):
            # if below the range, then continue until you get there
            if self.is_in_bound() < 0:
                # if you're below - then you need to add a new set
                if self.curr_var_ind < self.inj_level:
                    self.add_new_set()
                
                # update the coord of the loop before continuing looping
                self.update_coord()
                continue
                
            # run inner loop
            self.run_loop()

            # update the coord of the current loop_var, check if OOB
            if self.update_coord():
                break
        
        # go to one loop outer 
        self.dec_curr_var()
        
        # if you make it here - then you just return False
        return False
    
    def insert_spatial(self):
        timed_sites = self.get_all_timed_sites()
        # if output injection - don't need to expand spatially
        if self.WITH_SPATIAL or self.d_type == 'o':
            return timed_sites
        
        # timed_sites[i][j][k]
        # i = mem level
        # j = discrete time groups (i.e. separated by value being removed or overwritten)
        # k = possibilities within a time group
        
        if self.d_type == 'w':

            # loop through each mem_level
            for i in range(len(self.mem_inds)):
                mem_ind_unx, mem_ind = self.mem_inds[i]
                # grab the sites for the current level
                timed_site = timed_sites[i]
                # loop through the spatials - inside out (reverse order)
                for j in range(len(self.spatial_inds)-1, -1, -1):
                    # get the spatial ind, with form (ind, output_ind)
                    sp_ind, spunx_ind, out_ind, sp_type = self.spatial_inds[j]

                    # for weight, only have to handle parallel q, p
                    if sp_type not in ['q', 'p']:
                        continue

                    # get info for this spatial
                    # sp_size:  total size of the var (product of all consts = sp_inc*sp_const*sp_steps)
                    # sp_inc:   increment of each spatial (product of all consts below)
                    # sp_steps: number of times this spatial is called (product of all consts above)
                    # sp_const: const of the current spatial

                    # ex.-------------------------------------------------------
                    # for r [0:16]
                    #   pfor r [0:8]
                    #     for r [0:4]

                    # -->   sp_size     = 4*8*16
                    #       sp_inc      = 4
                    #       sp_steps    = 16
                    #       sp_const    = 8
                    sp_size, sp_inc, sp_steps, sp_const = self.spatial_info[j]

                    # loop through each timed group
                    for k in range(len(timed_site)):
                        timed_group = timed_site[k]
                        # print("timed: " + str(timed_group))
                        new_group = []
                        
                        # loop through each timed sites
                        for l in range(len(timed_group)):
                            # get the site iterated to and its length
                            single_site = timed_group[l]
                            num_single = len(single_site)
                            # array used if spunx_ind >= i --> duplicate into big group
                            big_site = [] # TODO: MIGHT NEED TO CHANGE TO BE BELOW
                            # loop through all increments needed (so loops through the const value)
                            for o in range(sp_const):
                                # array used if spunx_ind < i --> duplicate into dif groups
                                new_site = [] # TODO: MIGHT NEED TO CHANGE TO BE BELOW (currently 
                                            #       this will create behavior according to FIdelity) 
                                # iterate through number of locs in site
                                for m in range(num_single):
                                    # get one location (m, q, p)
                                    loc = single_site[m]
                                    # get the value of the spatial - so you can think of n as the 
                                    # index into the sp_steps (so product of all outside spatial)
                                    loc_val = loc[out_ind]
                                    n = loc_val//sp_inc     # get the index of the sites within one spatial element
                                    n_ = loc_val%sp_inc     # get the offset into that index

                                    # get the offset to index into
                                    off = n*sp_inc*sp_const

                                    new_val = 0
                                    new_val = off + o*sp_inc + n_

                                    if new_val < 0:
                                        continue

                                    # based on the out_ind (m, q, p)
                                    if out_ind == 0:
                                        new_loc = (new_val, loc[1], loc[2])
                                    elif out_ind == 1:
                                        new_loc = (loc[0], new_val, loc[2])
                                    else:
                                        new_loc = (loc[0], loc[1], new_val)
                                        
                                    # compare the spatial to the mem level
                                    # if the spatial is outside the mem_level
                                    if spunx_ind < mem_ind_unx:
                                        # need to duplicate into new/separate groups
                                        new_site.append(new_loc)
                                    # if the spatial is inside the mem level
                                    else:
                                        # duplicate into existing group
                                        big_site.append(new_loc)
                            
                                # add SITE TO NEW GROUP
                                # if spunx_ind < i:
                                if spunx_ind < mem_ind_unx and new_site:
                                    new_group.append(new_site)

                            if spunx_ind >= mem_ind_unx:
                                new_group.append(big_site)
                        
                        timed_site[k] = new_group
            return timed_sites
        
        elif self.d_type == 'i':
            # clear_log()
            # loop through each mem_level
            for i in range(len(self.mem_inds)):
                mem_ind_unx, mem_ind = self.mem_inds[i]
                # grab the sites for the current level
                timed_site = timed_sites[i]
                qs_proc = False
                pr_proc = False
                # loop through the spatials
                for j in range(len(self.spatial_inds)):
                    
                    # get the spatial ind, with form (ind, output_ind)
                    sp_ind, spunx_ind, out_ind, sp_type = self.spatial_inds[j]
                    
                    # only spatial for m, q, p
                    # if sp_type not in ['m', 'q', 'p']:
                    #     continue
                    if sp_type not in ['m', 'q', 'p', 's', 'r']:
                        continue
                    
                    proc_qs = sp_type in ['q', 's']
                    proc_pr = sp_type in ['p', 'r']
                    if (proc_qs and qs_proc) or (proc_pr and pr_proc):
                        continue
                    
                    qs_proc = proc_qs
                    pr_proc = proc_pr

                    # get info for this spatial
                    sp_size, sp_inc, sp_steps, sp_const = self.spatial_info[j]

                    # loop through each timed group
                    for k in range(len(timed_site)):
                        timed_group = timed_site[k]
                        # print("timed: " + str(timed_group))
                        new_group = []
                        
                        # loop through each timed sites
                        for l in range(len(timed_group)):
                            # get the site iterated to and its length
                            single_site = timed_group[l]
                            num_single = len(single_site)
                            # array used if spunx_ind >= i --> duplicate into big group
                            big_site = [] # TODO: MIGHT NEED TO CHANGE TO BE BELOW

                            # loop through all increments needed (so loops through the const value)
                            o_to_loop = []
                            if proc_qs:
                                o_to_loop = self.original_window[1]
                            elif proc_pr:
                                o_to_loop = self.original_window[2]
                            else:
                                o_to_loop = range(sp_const) # used to be just this
                            
                            
                            def dup_sites(sites, g_num, const, inc, out_ind, indiv=False):
                                out_set = []
                                curr_set = []
                                off = g_num*inc*const
                                for i in range(const):
                                    if indiv and curr_set:
                                        out_set.append(curr_set)
                                        curr_set = []
                                        
                                    for j in range(len(sites)):
                                        loc = sites[j]
                                        loc_val = loc[out_ind]
                                        n = loc_val % inc
                                        new_val = off + i*inc + n
                                        
                                        if out_ind == 0:
                                            new_loc = (new_val, loc[1], loc[2])
                                        elif out_ind == 1:
                                            new_loc = (loc[0], new_val, loc[2])
                                        else:
                                            new_loc = (loc[0], loc[1], new_val)
                                        
                                        curr_set.append(new_loc)
                                    
                                if curr_set:
                                    out_set.append(curr_set)
                                        
                                return out_set
                            
                            if out_ind == 0:
                                group_num = 0
                                sites_to_dup = []
                                    
                                # for o in o_to_loop: # put this first for eyeriss
                                for m in range(num_single):
                                    
                                    loc = single_site[m]
                                    loc_val = loc[out_ind]
                                    n = loc_val//sp_inc
                                    
                                    if n == group_num:
                                        pass
                                    else:
                                        to_indiv = spunx_ind < mem_ind_unx
                                        duped_sites = dup_sites(sites_to_dup, group_num, sp_const, sp_inc, out_ind, to_indiv)
                                        if duped_sites:
                                            if to_indiv:
                                                new_group += duped_sites
                                            else:
                                                big_site += duped_sites[0]
                    
                                        group_num = n
                                        sites_to_dup = []
                                        
                                    sites_to_dup.append(loc)

                                to_indiv = spunx_ind < mem_ind_unx
                                duped_sites = dup_sites(sites_to_dup, group_num, sp_const, sp_inc, out_ind, to_indiv)
                                if duped_sites:
                                    if to_indiv:
                                        new_group += duped_sites
                                    else:
                                        big_site += duped_sites[0]
                                        
                                if big_site and not spunx_ind < mem_ind_unx:
                                    new_group.append(big_site)
                            else:
                                for o in o_to_loop: # put this first for eyeriss
                                    # array used if spunx_ind < i --> duplicate into dif groups
                                    new_site = []   # TODO: MIGHT NEED TO CHANGE TO BE BELOW (currently 
                                                    #       this will create behavior according to FIdelity)

                                    # iterate through number of locs in site
                                    for m in range(num_single):
                                        # get one location (m, q, p)
                                        loc = single_site[m]

                                        new_val = o
                                        if new_val < 0:
                                            continue
                                        

                                        # based on the out_ind (m, q, p) insert new value
                                        if out_ind == 0:
                                            new_loc = (new_val, loc[1], loc[2])
                                        elif out_ind == 1:
                                            new_loc = (loc[0], new_val, loc[2])
                                        else:
                                            new_loc = (loc[0], loc[1], new_val)

                                        # compare the spatial to the mem level
                                        # if the spatial is outside the mem_level
                                        if spunx_ind < mem_ind_unx:
                                            # need to duplicate into new/separate groups
                                            new_site.append(new_loc)
                                        # if the spatial is inside the mem level
                                        else:
                                            # duplicate into existing group
                                            big_site.append(new_loc)
                                
                                    # add SITE TO NEW GROUP
                                    if spunx_ind < mem_ind_unx and new_site:
                                        new_group.append(new_site)

                                # if inside mem_level - then add the big site
                                if spunx_ind >= mem_ind_unx:
                                    new_group.append(big_site)
                        
                        timed_site[k] = new_group
                                        
            return timed_sites
        elif self.d_type == 'o':
            pass
        else:
            # should never get here
            assert(False)
            
    def prune_sites(self, sites):
        ranges = self.original_window
        new_sites = []
        # print("Pruning into window = " + str(ranges))
        for i in range(len(sites)):
            sites0 = sites[i]
            new_sites0 = []
            for j in range(len(sites0)):
                sites1 = sites0[j]
                new_sites1 = []
                for k in range(len(sites1)):
                    sites2 = sites1[k]
                    new_sites2 = []
                    for l in range(len(sites2)):
                        site = sites2[l]
                        is_in = (site[0] in ranges[0]) and (site[1] in ranges[1]) and (site[2] in ranges[2])
                        if is_in:
                            new_sites2.append(site)
                    if new_sites2:
                        new_sites1.append(new_sites2)
                if new_sites1:
                    new_sites0.append(new_sites1)
            if new_sites0:
                new_sites.append(new_sites0)
                            
        return new_sites
    
# W = weight kernel size
# S = stride
# I = target index value
# L = input size
def get_window_i(W, S, I, L):
    o = 0
    i = 0
    ranges = []
    in_range = False
    while True:
        # if out of bounds, stop
        if i + W > L:
            ranges.append(o)
            break

        # loop until you get in range
        if I in range(i, i+W):
            # if first time in range - append beginning of window
            if not in_range:
                in_range = True
                ranges.append(o)
        # if out of range
        else:
            # if you were in range - append end of window
            if in_range:
                ranges.append(o)
                break
        o += 1
        i += S

        assert(i < 1000)
    try:
        out_range = range(ranges[0], ranges[1])
    except:
        print(ranges)
        print("Weight width: %d\nStride: %d\nTarget input index: %d\nInput size: %d"%(W, S, I, L))
        assert(False)
        
    return out_range

# var_sizes will have the form [m, c, s, r, q, p, h, w]
def get_window(inj_ind, var_sizes, strides=[1,1], padding=[0,0], d_type='i'):
    assert(len(var_sizes) == 8)
    m, c, s, r, q, p, h, w = var_sizes
    h += padding[0]
    w += padding[1]
    
    # if i - do quick calculation to get the window
    if d_type == 'i':
        return (range(m), get_window_i(s, strides[0], inj_ind[1], h), get_window_i(r, strides[1], inj_ind[2], w))
    # if w - then it's just the whole output channel (of the m)
    elif d_type == 'w':
        m = inj_ind[0]
        return (range(m, m+1), range(q), range(p))
    # if o - then it's just the exact inj_ind - so just turn that into ranges
    elif d_type == 'o':
        inj_m, inj_q, inj_p = inj_ind
        return (range(inj_m, inj_m+1), range(inj_q, inj_q+1), range(inj_p, inj_p+1),)
    else:
        assert(False)
        
# var_sizes will have the form [m, c, s, r, q, p, h, w]
def check_sites(sites, inj_ind, var_sizes, strides, padding=[0,0], d_type='i'):
    assert(len(var_sizes) == 8)
    
    ranges = get_window(inj_ind, var_sizes, strides, padding=padding, d_type=d_type)
        
    for site in sites:
        is_in = site[0] in ranges[0] and site[1] in ranges[1] and site[2] in ranges[2] 
        if not is_in:
            print(site)
            print(ranges)
            assert(False)