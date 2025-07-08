/* 2, 202428015059018, WangAiLing */

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "GraphLite.h"
#include <algorithm>
#include <time.h>
#include <stdlib.h>
#define VERTEX_CLASS_NAME(name) GraphColor##name

int m_v0_id;
int m_num_color;
//#define EPS 1e-6

class VERTEX_CLASS_NAME(InputFormatter): public InputFormatter {
public:
    int64_t getVertexNum() {
        unsigned long long n;
        sscanf(m_ptotal_vertex_line, "%lld", &n);
        m_total_vertex= n;
        return m_total_vertex;
    }
    int64_t getEdgeNum() {
        unsigned long long n;
        sscanf(m_ptotal_edge_line, "%lld", &n);
        m_total_edge= n;
        return m_total_edge;
    }
    int getVertexValueSize() {
        m_n_value_size = sizeof(int);
        return m_n_value_size;
    }
    int getEdgeValueSize() {
        m_e_value_size = sizeof(double);
        return m_e_value_size;
    }
    int getMessageValueSize() {
        m_m_value_size = sizeof(int);
        return m_m_value_size;
    }
    void loadGraph() {
        unsigned long long last_vertex;
        unsigned long long from;
        unsigned long long to;
        // No weight
        double weight = 0;

        int value = -1;
        int outdegree = 0;

        const char *line= getEdgeLine();

        // Note: modify this if an edge weight is to be read
        //       modify the 'weight' variable

        sscanf(line, "%lld %lld", &from, &to);
        addEdge(from, to, &weight);

        last_vertex = from;
        ++outdegree;
        for (int64_t i = 1; i < m_total_edge; ++i) {
            line= getEdgeLine();

            // Note: modify this if an edge weight is to be read
            //       modify the 'weight' variable

            sscanf(line, "%lld %lld", &from, &to);
            if (last_vertex != from) {
                addVertex(last_vertex, &value, outdegree);
                last_vertex = from;
                outdegree = 1;
            } else {
                ++outdegree;
            }
            addEdge(from, to, &weight);
        }
        addVertex(last_vertex, &value, outdegree);
    }
};

class VERTEX_CLASS_NAME(OutputFormatter): public OutputFormatter {
public:
    void writeResult() {
        int64_t vid;
        int color = -1;
        char s[1024];

        for (ResultIterator r_iter; ! r_iter.done(); r_iter.next() ) {
            r_iter.getIdValue(vid, &color);
            int n = sprintf(s, "%lld: %d\n", (unsigned long long)vid, color);
            writeNextResLine(s, n);
        }
    }
};

class VERTEX_CLASS_NAME(): public Vertex <int, double, int> {
public:
    void compute(MessageIterator* pmsgs) {
        int step = getSuperstep();
        int vertex_id = getVertexId();
        if (step == 0)
            if (vertex_id == m_v0_id) { // Special case for the starting vertex
               * mutableValue() = 0;
               sendMessageToAllNeighbors(0);
               voteToHalt();
               return;
            }


        bool tmp[m_num_color];
        for (int i=0;i<m_num_color;++i)
            tmp[i] = false;
        int tmp_cnt = 0;
//
        for ( ; ! pmsgs->done(); pmsgs->next() ) {
            int color = pmsgs->getValue();
            if (tmp[color] == false) {
                tmp_cnt++;
                tmp[color] = true;
            }
        }

        if (tmp_cnt == 0) { // No message
            voteToHalt();
            return;
        }

        int oldColor = getValue();

        int newColor = (oldColor == -1 ? 0 : oldColor);

        while (tmp[newColor]) {
            newColor = rand() % m_num_color;
        }

        if (newColor != oldColor){
            * mutableValue() = newColor;
            sendMessageToAllNeighbors(newColor);
        }

        voteToHalt();
    }
};

class VERTEX_CLASS_NAME(Graph): public Graph {

public:
    /*
    command line:
       $ start-graphlite example/your_program.so <input path> <output path> <v0 id> <num color>
    */
    // argv[0]: PageRankVertex.so
    // argv[1]: <input path>
    // argv[2]: <output path>
    // argv[3]: <v0 id>
    // argv[4]: <num color>

    void init(int argc, char* argv[]) {
        srand(time(NULL));

        setNumHosts(5);
        setHost(0, "localhost", 1411);
        setHost(1, "localhost", 1421);
        setHost(2, "localhost", 1431);
        setHost(3, "localhost", 1441);
        setHost(4, "localhost", 1451);
        // Check input arguments
        if (argc < 5) {
            printf ("Usage: %s <input path> <output path> <start vertex> <color num>\n", argv[0]);
            exit(1);
        }

        m_pin_path = argv[1];
        m_pout_path = argv[2];
        m_v0_id = atoi(argv[3]);
        m_num_color = atoi(argv[4]);

    }


};

/* STOP: do not change the code below. */
extern "C" Graph* create_graph() {
    Graph* pgraph = new VERTEX_CLASS_NAME(Graph);

    pgraph->m_pin_formatter = new VERTEX_CLASS_NAME(InputFormatter);
    pgraph->m_pout_formatter = new VERTEX_CLASS_NAME(OutputFormatter);
    pgraph->m_pver_base = new VERTEX_CLASS_NAME();

    return pgraph;
}

extern "C" void destroy_graph(Graph* pobject) {
    delete ( VERTEX_CLASS_NAME()* )(pobject->m_pver_base);
    delete ( VERTEX_CLASS_NAME(OutputFormatter)* )(pobject->m_pout_formatter);
    delete ( VERTEX_CLASS_NAME(InputFormatter)* )(pobject->m_pin_formatter);
    delete ( VERTEX_CLASS_NAME(Graph)* )pobject;
}
