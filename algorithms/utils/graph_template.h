

template<typename indexType>
struct My_Graph{

    using iT = indexType; //necessary to recover indexType in some contexts

    struct edgeRange{ //struct for managing updates to a single neighbor

        size_t size() const {} //return exact size of edge list

        indexType id() const {} //return id of edgeRange

        edgeRange() {} //default constructor (mandatory)

        edgeRange(args A) {} //constructor 

        indexType operator [] (indexType j) const {} //returns id of jth element of edge list

        void clear(){} //removes all neighbors

        template<typename rangeType>
        void update_neighbors(const rangeType& r){} //replaces edge list with edges contained in r

        template<typename rangeType>
        void append_neighbors(const rangeType& r){} //appends ids in r to edge list

        void prefetch(){} //optional but can help with performance

        // template<typename F>
        // void sort(F&& less){std::sort(edges.begin()+1, edges.begin()+1+edges[0], less);}

        template<typename F>
        void reorder(F&& f){} //OPTIONAL reorder edge list according to lambda f

        parlay::slice<indexType*, indexType*> neighbors(){} //return iterable range for edge list
            
    }; //end Edge_Range


    struct Graph{ //must be named Graph
        long max_degree() const {} //return max degree

        size_t size() const {} //return exact number of vertices

        Graph(){} //default constructor

        Graph(long maxDeg, size_t n) {} //constructor with size and maxDeg only

        Graph(char* gFile){} //load graph from file

        void save(char* oFile){} //save graph to file

        //takes in a sequence of pairs of indextype and edge lists and sets the edge list of index i to those edges
        void batch_update(parlay::sequence<std::pair<indexType, parlay::sequence<indexType>>> &edges){}

        //nb currently optional for graph building
        void batch_delete(parlay::sequence<indexType> deletes){} //takes in a range of indices to delete and deletes them from the graph


        edgeRange operator [] (indexType i) {} //returns an edgeRange data structure
  
    }; //end Graph


    //next 3 functions are graph constructors
    My_Graph(){}

    My_Graph(long maxDeg, size_t n) {}

    My_Graph(char* gFile){G = Graph(gFile);}

    //see vamana/index.h and utils/beamSearch.h for use of these functions

    //next 2 functions are getters for the underlying graph
    //the read_only distinction only matters for versioned graphs
    Graph& Get_Graph() {return G;}

    Graph& Get_Graph_Read_Only() {return G;}

    //Release_Graph() should be called at the end of a block started by Get_Graph_Read_Only()
    void Release_Graph(Graph GR){}

    //Update_Graph() should be called on a new graph created after a call to Get_Graph()
    void Update_Graph(Graph GR){G = GR;}

    void save(char* oFile){G.save(oFile);}

    private:
        Graph G;

}; //end Flat_Graph
