module Model

struct Config
    lr          ::Float64
    epochs      ::Int
    batch_size  ::Int
end

function train(data, cfg::Config)
    for epoch = 1:cfg.epochs
        loss = sum(x^2 for x in data)/length(data)
        @info "epoch" epoch=epoch loss=loss
    end
end

function evaluate(preds, targets; metric=:mse)
    if metric==:mse
        sum((p-t)^2 for (p,t) in zip(preds,targets))/length(preds)
    elseif metric==:mae
        sum(abs(p-t) for (p,t) in zip(preds,targets))/length(preds)
    else
        error("unknown metric: $(metric)")
    end
end

end
