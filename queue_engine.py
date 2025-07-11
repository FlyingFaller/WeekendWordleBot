from helpers import *
from multiprocessing import shared_memory
import time

@njit(cache=True)
def _get_candidates(pattern_matrix, nguesses, nanswers, ans_idxs, nprune):
    pattern_data = []
    log2_nanswers = np.log2(nanswers)
    entropy_vals = np.zeros(nguesses, dtype=np.float64)
    pattern_columns = pattern_matrix[:, ans_idxs]

    for i in range(nguesses):
        pattern_row = pattern_columns[i]
        all_pcounts = np.bincount(pattern_row, minlength=243)
        patterns = np.nonzero(all_pcounts)[0]

        if len(patterns) < 2:
            pattern_data.append(None)
            continue

        pcounts = all_pcounts[patterns]
        pattern_data.append((patterns, pcounts))
        sum_c_log2_c = np.sum(pcounts * np.log2(pcounts))
        entropy_vals[i] = log2_nanswers - (sum_c_log2_c / nanswers)

    candidate_idxs = np.argsort(entropy_vals)[-nprune:]
    return (candidate_idxs, pattern_data, pattern_columns)

class EvaluationWorker(multiprocessing.Process):
    """
    A worker that pulls evaluation tasks, computes entropy and next steps,
    and queues up new evaluation or aggregation tasks.
    """
    # def __init__(self, task_queue, agg_queues, results_dict, pattern_matrix, nguesses, nprune):
    #     super().__init__()
    #     self.task_queue = task_queue
    #     self.agg_queues = agg_queues
    #     self.results_dict = results_dict
    #     self.pattern_matrix = pattern_matrix
    #     self.nguesses = nguesses
    #     self.nprune = nprune
    #     self.recieived_sentinel = False
    def __init__(self, task_queue, agg_queues, results_dict, nguesses, nprune, shm_name, matrix_shape, matrix_dtype):
        super().__init__()
        self.task_queue = task_queue
        self.agg_queues = agg_queues
        self.results_dict = results_dict
        self.nguesses = nguesses
        self.nprune = nprune
        
        # Store the info needed to connect to the shared memory
        self.shm_name = shm_name
        self.matrix_shape = matrix_shape
        self.matrix_dtype = matrix_dtype
        self.pattern_matrix = None # Will be connected in run()

    def run(self):
        existing_shm = shared_memory.SharedMemory(name=self.shm_name)
        
        # Create a NumPy array that uses the shared memory buffer
        self.pattern_matrix = np.ndarray(
            self.matrix_shape, dtype=self.matrix_dtype, buffer=existing_shm.buf
        )
    
        while True:
            task = self.task_queue.get()
            # --- Main processing logic ---
            try:
                if task is None:
                    # Signal that this worker has finished its part of the queue
                    break

                task_id, ans_idxs, nanswers, current_depth = task

                if task_id in self.results_dict:
                    continue

                self.results_dict[task_id] = -1.0

                # pattern_data = []
                # log2_nanswers = np.log2(nanswers)
                # entropy_vals = np.zeros(self.nguesses, dtype=np.float64)
                # # pattern_columns = self.pattern_matrix[:, ans_idxs]
                # pattern_columns = self.pattern_matrix[:, ans_idxs]

                # for i in range(self.nguesses):
                #     pattern_row = pattern_columns[i]
                #     all_pcounts = np.bincount(pattern_row, minlength=243)
                #     patterns = np.nonzero(all_pcounts)[0]

                #     if len(patterns) < 2:
                #         pattern_data.append(None)
                #         continue

                #     pcounts = all_pcounts[patterns]
                #     pattern_data.append((patterns, pcounts))
                #     sum_c_log2_c = np.sum(pcounts * np.log2(pcounts))
                #     entropy_vals[i] = log2_nanswers - (sum_c_log2_c / nanswers)

                # candidate_idxs = np.argsort(entropy_vals)[-self.nprune:]

                candidate_idxs, pattern_data, pattern_columns = _get_candidates(self.pattern_matrix, self.nguesses, nanswers, ans_idxs, self.nprune)

                ##### REFINED IMPLEMENTATION #####
                candidates_to_aggregate = []
                min_resolved_score = np.inf
                for candidate_idx in candidate_idxs:
                    patterns, pcounts = pattern_data[candidate_idx]
                    pattern_row = pattern_columns[candidate_idx]

                    child_tasks = []
                    child_dependencies = []
                    partial_score = 0.0
                    resolved = True

                    # Scoring logic
                    for pattern, count in zip(patterns, pcounts):
                        if pattern == 242:
                            continue
                        elif count < 3:
                            partial_score += 2 * count - 1
                        else:
                            child_ans_idxs = ans_idxs[pattern_row == pattern]
                            # child_id = FNV_hash(child_ans_idxs)
                            child_id = python_hash(child_ans_idxs)
                            # child_id = blake2b(child_ans_idxs)
                            child_score = self.results_dict.get(child_id, None)
                            # if child_id in self.results_dict and self.results_dict[child_id] != -1:
                            if child_score is not None and child_score != -1:
                                partial_score += self.results_dict[child_id]
                            else:
                                child_tasks.append((child_id, child_ans_idxs, count, current_depth + 1))
                                child_dependencies.append((child_id, count))
                                resolved = False

            
                    score = (partial_score / nanswers) + 1
                    if score < min_resolved_score:
                        if resolved:
                            min_resolved_score = score
                        else:
                            # We cannot be sure the partial score will not be the min score so we create the tasks and add it to be aggregated
                            for child_task in child_tasks:
                                self.task_queue.put(child_task)

                            candidates_to_aggregate.append({
                                "children": child_dependencies,
                                "partial_score": partial_score
                            })
                
                if not candidates_to_aggregate:
                    self.results_dict[task_id] = min_resolved_score
                else: 
                    agg_task = {
                        "parent_id": task_id,
                        "candidates_to_agg": candidates_to_aggregate,
                        "min_resolved_score": min_resolved_score,
                        "nanswers": nanswers,
                        "failed": False
                    }

                    self.agg_queues[current_depth].put(agg_task)

            finally:
                # This is crucial: signal that the task we got is done.
                self.task_queue.task_done()


# --- Phase 2: Aggregation Worker ---

class AggregationWorker(multiprocessing.Process):
    """
    A worker that takes completed aggregation tasks from a single queue
    and calculates the final score for parent nodes.
    """
    def __init__(self, agg_queues, results_dict, barrier):
        super().__init__()
        self.agg_queues = agg_queues
        self.results_dict = results_dict
        self.barrier = barrier

    def run(self):
        for i, agg_queue in enumerate(self.agg_queues):
            while True:
                task = agg_queue.get()
                try:
                    if task is None:
                        break

                    parent_id = task['parent_id']
                    candidates_to_agg = task['candidates_to_agg']
                    min_score = task['min_resolved_score']
                    nanswers = task['nanswers']
                    failed = task['failed']

                    if self.results_dict[parent_id] != -1:
                        continue

                    continue_while = False

                    for candidate in candidates_to_agg:
                        partial_score = candidate['partial_score']

                        for child_id, count in candidate['children']:
                            child_score = self.results_dict[child_id]
                            if child_score == -1.0:
                                if not failed:
                                    print(f"Failed to resolve child {child_id} of parent {parent_id} once")
                                    agg_queue.put({'parent_id': task['parent_id'],
                                                   'candidates_to_agg': task['candidates_to_agg'],
                                                   'min_resolved_score': task['min_resolved_score'],
                                                   'nanswers': task['nanswers'],
                                                   'failed': True})
                                    continue_while = True
                                    break
                                else:
                                    print(f"Failed to resolve child {child_id} of parent {parent_id} twice!")
                                    if i+1 < len(self.agg_queues):
                                        self.agg_queues[i+1].put({'parent_id': task['parent_id'],
                                                                  'candidates_to_agg': task['candidates_to_agg'],
                                                                  'min_resolved_score': task['min_resolved_score'],
                                                                  'nanswers': task['nanswers'],
                                                                  'failed': False})
                                        continue_while = True
                                        break
                                    else:
                                        continue
                                
                            partial_score += child_score * count

                                                # If the flag is set, break out of the middle loop
                        if continue_while:
                            break

                        score = (partial_score / nanswers) + 1

                        if score < min_score:
                            min_score = score

                    if continue_while:
                        continue    
                        
                    self.results_dict[parent_id] = min_score

                finally: 
                    agg_queue.task_done()

            # print(f"Worker {self.pid} waiting at barrier...")
            self.barrier.wait()

# --- Orchestrator ---

def run_solver(pattern_matrix, initial_ans_idxs, nguesses, nprune, max_depth, num_workers):
    """
    Manages the two-phase process of evaluation and aggregation.
    """
    # --- 1. Create the Shared Memory Block ---
    shm = shared_memory.SharedMemory(create=True, size=pattern_matrix.nbytes)
    
    # Create a new NumPy array that uses the shared memory buffer
    shared_matrix = np.ndarray(pattern_matrix.shape, dtype=pattern_matrix.dtype, buffer=shm.buf)
    
    # Copy your matrix data into the shared memory
    shared_matrix[:] = pattern_matrix[:]

    with multiprocessing.Manager() as manager:
        try:
            results_dict = manager.dict()
            # CHANGE: Use JoinableQueue for robust termination detection
            s1 = time.time()
            evaluation_queue = multiprocessing.JoinableQueue()
            aggregation_queues = [manager.JoinableQueue() for _ in range(max_depth + 1)]
            s2 = time.time()
            print(f"{'Queue creation took':.<40}{s2-s1:.6f} sec")

            print("\n--- Starting Phase 1: Evaluation ---\n")
            s1 = time.time()
            eval_workers = [
                EvaluationWorker(
                    evaluation_queue,
                    aggregation_queues, # agg_queues not needed in this snippet
                    results_dict,
                    nguesses,
                    nprune,
                    shm.name, # Pass the name, not the whole object
                    pattern_matrix.shape,
                    pattern_matrix.dtype
                )
                for _ in range(num_workers)
            ]
            s2 = time.time()
            print(f"{'Worker creation took':.<40}{s2-s1:.6f} sec")

            s1 = time.time()
            for worker in eval_workers:
                worker.start()
            s2 = time.time()
            print(f"{'Workers started in':.<40}{s2-s1:.6f} sec")

            s1 = time.time()
            initial_nanswers = len(initial_ans_idxs)
            # root_id = FNV_hash(initial_ans_idxs)
            root_id = python_hash(initial_ans_idxs)
            # root_id = blake2b(initial_ans_idxs)
            initial_task = (root_id, initial_ans_idxs, initial_nanswers, 0)
            evaluation_queue.put(initial_task)
            s2 = time.time()
            print(f"{'Task seeded in':.<40}{s2-s1:.6f} sec")

            s1 = time.time()
            print("Main thread waiting for evaluation tasks to complete...")
            evaluation_queue.join()
            s2 = time.time()
            print(f"{'All tasks processed in':.<40}{s2 -s1:.6f} sec")

            # Send stop signals (sentinel values) to workers
            s1 = time.time()
            for _ in range(num_workers):
                evaluation_queue.put(None)
            s2 = time.time()
            print(f'{'Sentinels added in':.<40}{s2 - s1:.6f} sec')

            s1 = time.time()
            evaluation_queue.close()
            evaluation_queue.join_thread()
            s2 = time.time()
            print(f'{'Closed the queue in':.<40}{s2 - s1:.6f} sec')

            s1 = time.time()
            for worker in eval_workers:
                worker.join()
            s2 = time.time()
            print(f'{'Workers all shut down in':.<40}{s2-s1:.6f} sec')

            print("\n--- Starting Phase 2: Aggregation ---\n")
            barrier = multiprocessing.Barrier(num_workers)

            queues_in_order = [aggregation_queues[d] for d in range(max_depth, -1, -1)]

            s1 = time.time()
            agg_workers = [
                AggregationWorker(queues_in_order, results_dict, barrier)
                for _ in range(num_workers)
            ]
            s2 = time.time()
            print(f'{'Workers created in':.<40}{s2-s1:.6f} sec')

            s1 = time.time()
            for worker in agg_workers:
                worker.start()
            s2 = time.time()
            print(f"{'Workers started in':.<40}{s2-s1:.6f} sec")

            print('Waiting for all queues to join...')
            s1 = time.time()
            for i, q in enumerate(queues_in_order):
                # Wait for all tasks in the CURRENT queue to be processed
                print(f"...Approximate queue size at depth {max_depth-i}: {q.qsize()}")

                q.join()

                # Put sentinels on the CURRENT queue to release the workers
                # from their inner loop, so they can hit the barrier.
                for _ in range(num_workers):
                    q.put(None)
            s2 = time.time()
            print(f'{'Queues complete and sentinels sent in':.<40}{s2-s1:.6f} sec')

            # 5. After all queues are done, join the worker processes
            s1 = time.time()
            for worker in agg_workers:
                worker.join()
            s2 = time.time()
            print(f"{'Workers all shut down in':.<40}{s2-s1:.6f} sec")

            final_result = results_dict.get(root_id, "Result not found.")
            print(f"\nFinal Result for root node: {final_result}")
            print(f"Cache size: {len(results_dict)}")
            return final_result
        
        finally:
            print("\nCleaning up shared memory...")
            shm.close()
            shm.unlink() # Frees the memory block