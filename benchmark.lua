wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '{"text": "Aristotle was the smartest of his time"}'

-- Ensure SSL works properly
done = function(summary, latency, requests)
    print("\n=== Benchmark Results ===")
    print("Total Requests: ", summary.requests)
    print("Total Errors: ", summary.errors.status + summary.errors.connect + summary.errors.timeout)
    print("Requests Per Second (RPS): ", summary.requests / (summary.duration / 1000000))
    print("Average Latency: ", latency.mean / 1000, "ms")
    print("p50 Latency: ", latency:percentile(50.0) / 1000, "ms")
    print("p95 Latency: ", latency:percentile(95.0) / 1000, "ms")
    print("p99 Latency: ", latency:percentile(99.0) / 1000, "ms")
end
