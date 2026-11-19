# Unit II – Process Management (Lecture Notes)

## 1. Process Concept
A **process** is a program in execution. A program is a passive entity (an executable file stored on disk), whereas a process is an active entity with a program counter specifying the next instruction to execute and a set of associated resources. Two processes may be associated with the same program, yet they are considered separate execution sequences.

A process in memory consists of the text section (program code), the data section (global variables), the heap (dynamically allocated memory) and the stack (function parameters, return addresses, local variables).

### Process States
As a process executes, it changes state:
- **New** – the process is being created.
- **Ready** – the process is waiting to be assigned to a processor.
- **Running** – instructions are being executed.
- **Waiting** – the process is waiting for some event (such as I/O completion).
- **Terminated** – the process has finished execution.

Transitions: New → Ready (admitted), Ready → Running (scheduler dispatch), Running → Ready (interrupt), Running → Waiting (I/O or event wait), Waiting → Ready (I/O or event completion), Running → Terminated (exit). Only one process can be running on any processor core at any instant.

### Process Control Block (PCB)
Each process is represented in the operating system by a **Process Control Block**, also called a task control block. The PCB contains:
- Process state
- Program counter
- CPU registers
- CPU-scheduling information (priority, pointers to scheduling queues)
- Memory-management information (base and limit registers, page tables)
- Accounting information (CPU time used, time limits, process numbers)
- I/O status information (list of I/O devices allocated, open files)

### Context Switch
Switching the CPU to another process requires saving the state of the current process in its PCB and restoring the saved state of a different process. This task is known as a **context switch**. Context-switch time is pure overhead, because the system does no useful work while switching. Its speed depends on memory speed, the number of registers to be copied, and hardware support.

## 2. Threads
A **thread** is a basic unit of CPU utilization; it comprises a thread ID, a program counter, a register set and a stack. It shares with other threads of the same process its code section, data section and other OS resources such as open files.

### Benefits of Multithreading
1. **Responsiveness** – an application can continue running even if part of it is blocked.
2. **Resource sharing** – threads share the memory and resources of their process.
3. **Economy** – creating and context-switching threads is cheaper than processes.
4. **Scalability** – threads can run in parallel on multicore processors.

### Multithreading Models
- **Many-to-One model** – maps many user-level threads to one kernel thread. Thread management is done in user space, but the entire process blocks if one thread makes a blocking system call, and threads cannot run in parallel on multicore systems.
- **One-to-One model** – maps each user thread to a kernel thread. It provides more concurrency and allows parallel execution, but creating a user thread requires creating a kernel thread (overhead). Used by Linux and Windows.
- **Many-to-Many model** – multiplexes many user-level threads to a smaller or equal number of kernel threads. It combines the benefits of both; the two-level model is a variation that also allows a user thread to be bound to a kernel thread.

## 3. CPU Scheduling
**CPU scheduling** is the basis of multiprogrammed operating systems: the short-term scheduler selects a process from the ready queue and allocates the CPU to it.

Scheduling criteria:
- **CPU utilization** – keep the CPU as busy as possible.
- **Throughput** – number of processes completed per time unit.
- **Turnaround time** – interval from submission of a process to its completion.
- **Waiting time** – total time spent waiting in the ready queue.
- **Response time** – time from submission until the first response is produced.

### Scheduling Algorithms
- **First-Come, First-Served (FCFS)** – the process that requests the CPU first is allocated first; implemented with a FIFO queue. Non-preemptive. Suffers from the convoy effect. Example: P1=24, P2=3, P3=3 ms arriving in order gives average waiting time (0+24+27)/3 = 17 ms.
- **Shortest-Job-First (SJF)** – allocates the CPU to the process with the smallest next CPU burst. Optimal for minimum average waiting time. For the same example in order P2, P3, P1: (0+3+6)/3 = 3 ms. Preemptive SJF is called shortest-remaining-time-first.
- **Priority scheduling** – the CPU is allocated to the highest-priority process. Problem: starvation (indefinite blocking); solution: aging, i.e. gradually increasing the priority of waiting processes.
- **Round Robin (RR)** – designed for time-sharing; each process gets a small unit of CPU time called a time quantum (10–100 ms), after which it is preempted and added to the tail of the ready queue. Performance depends heavily on the quantum size.

## 4. Process Synchronization
### Critical-Section Problem
A **critical section** is a segment of code in which a process may change shared variables, update a table or write a file. When one process is executing in its critical section, no other process is allowed to execute in its critical section.

A solution must satisfy three requirements:
1. **Mutual exclusion** – only one process executes in its critical section at a time.
2. **Progress** – if no process is in its critical section, the selection of the next process cannot be postponed indefinitely.
3. **Bounded waiting** – there is a bound on the number of times other processes may enter their critical sections after a process has requested entry.

### Peterson's Solution
Peterson's solution is a classic software solution for two processes Pi and Pj. They share two variables: `int turn;` and `boolean flag[2];`. `turn` indicates whose turn it is to enter the critical section; `flag[i] = true` indicates Pi is ready to enter.

```
do {
    flag[i] = true;
    turn = j;
    while (flag[j] && turn == j);   // busy wait
        /* critical section */
    flag[i] = false;
        /* remainder section */
} while (true);
```
It satisfies mutual exclusion, progress and bounded waiting.

### Semaphores
A **semaphore** S is an integer variable that, apart from initialization, is accessed only through two atomic operations: `wait()` (P) and `signal()` (V).

```
wait(S)   { while (S <= 0); S--; }
signal(S) { S++; }
```
A counting semaphore can range over an unrestricted domain; a binary semaphore (mutex lock) ranges only between 0 and 1.

**Producer–Consumer (bounded buffer) using semaphores:** with n buffers, use `mutex = 1`, `empty = n`, `full = 0`.
- Producer: `wait(empty); wait(mutex); add item; signal(mutex); signal(full);`
- Consumer: `wait(full); wait(mutex); remove item; signal(mutex); signal(empty);`

## 5. Deadlocks
A **deadlock** is a situation in which a set of processes are each waiting for an event that can be caused only by another process in the set.

Four necessary conditions (all must hold simultaneously):
1. **Mutual exclusion** – at least one resource is non-sharable.
2. **Hold and wait** – a process holds at least one resource while waiting for others.
3. **No preemption** – resources cannot be forcibly taken away.
4. **Circular wait** – a circular chain of processes exists, each waiting for a resource held by the next.

### Deadlock Prevention
Ensure that at least one necessary condition cannot hold: make resources sharable where possible; require processes to request all resources at once (hold and wait); allow preemption of resources; impose a total ordering of resource types and require requests in increasing order (circular wait).

### Banker's Algorithm (Deadlock Avoidance)
Each process declares the maximum number of instances of each resource type it may need. Data structures: **Available**, **Max**, **Allocation**, **Need** (Need = Max − Allocation). The **safety algorithm** checks whether the system is in a safe state by finding a sequence in which every process can obtain its needed resources and finish. The **resource-request algorithm** grants a request only if the resulting state is safe; otherwise the process must wait.
