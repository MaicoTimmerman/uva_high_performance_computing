#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <mpi.h>

int MPI_Broadcast( void *buffer, /* INOUT : buffer address */
        int count, /* IN : buffer size */
        MPI_Datatype datatype, /* IN : datatype of entry */
        int root, /* IN : root process (sender) */
        MPI_Comm communicator) /* IN : communicator */ {

    int world_rank, world_size;
    MPI_Comm_rank(communicator, &world_rank);
    MPI_Comm_size(communicator, &world_size);

    if (world_rank == root) {
        // If we are the root process, send our data to everyone
        for (int i = 0; i < world_size; i++) {
            if (i != world_rank) {
                MPI_Send(buffer, count, datatype, i, 0, communicator);
            }
        }
        return MPI_SUCCESS;
    } else {
        // If we are a receiver process, receive the data from the root
        MPI_Recv(buffer, count, datatype, root, 0, communicator,
                MPI_STATUS_IGNORE);
        return MPI_SUCCESS;
    }
}

int MPI_Broadcast2( void *buffer, /* INOUT : buffer address */
        int count, /* IN : buffer size */
        MPI_Datatype datatype, /* IN : datatype of entry */
        int root, /* IN : root process (sender) */
        MPI_Comm communicator) /* IN : communicator */
{
    int world_rank, world_size;
    MPI_Comm_rank(communicator, &world_rank);
    MPI_Comm_size(communicator, &world_size);


    // If not root, receive the message from neighbour.
    if (world_rank != root) {
        MPI_Recv(buffer, count, datatype, (world_rank - 1) % world_size, 0, communicator,
                MPI_STATUS_IGNORE);
    }

    // Send to the next neighbour in the ring topology
    MPI_Send(buffer, count, datatype, (world_rank + 1) % world_size, 0,
            MPI_COMM_WORLD);

    // Receive from the other end,
    if (world_rank == root) {
        MPI_Recv(buffer, count, datatype, (world_rank - 1) % world_size, 0, communicator,
                MPI_STATUS_IGNORE);
    }

    return MPI_SUCCESS;
}


int main(int argc, char **argv) {
    int num_tasks, world_rank;

    if (argc < 2) {
        printf("Usage [%s] [comm_method].\n", argv[0]);
        printf("\t comm_method: 0 = many-to-many, 1 = ring-topology\n");
    }

    int communication_method = atoi(argv[1]);

    int rc = MPI_Init(&argc, &argv); // Initialize MPI runtime
    if (rc != MPI_SUCCESS) { // Check for success
        fprintf(stderr, "Unable to set up MPI\n");
        MPI_Abort(MPI_COMM_WORLD, rc); // Abort MPI runtime
    }


    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks); // Get num tasks
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get task id

    int data;

    // Data is only known to the root process
    if (world_rank == 0) {
        data = 1337;
    }

    switch (communication_method) {
        case 0:
            MPI_Broadcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);
            break;
        case 1:
            MPI_Broadcast2(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);
            break;
        default:
            printf("Unsupported cummuncation method\n");

    }

    printf("[%d] Data contains '%d'.\n", world_rank, data);

    return MPI_Finalize();
}
