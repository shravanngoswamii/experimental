module Model

using Statistics

struct Config
    lr          ::Float64
    epochs      ::Int
    batch_size  ::Int
end

function train(data, cfg::Config)
    loss_history = Float64[]
    for epoch = 1:cfg.epochs
        loss = 0.0
        for i = 1:cfg.batch_size:length(data)
            batch = data[i:min(i+cfg.batch_size-1, end)]
            loss += sum(x^2 for x in batch)
        end
        push!(loss_history, loss/cfg.epochs)
        @info "epoch done" epoch=epoch loss=loss
    end
    loss_history
end

function evaluate(model, test_data; metric=:mse)
    if metric==:mse
        mean(x^2 for x in test_data)
    elseif metric==:mae
        mean(abs(x) for x in test_data)
    else
        error("unknown metric: $(metric)")
    end
end

end
