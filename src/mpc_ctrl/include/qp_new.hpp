#ifndef QP_NEW_H
#define QP_NEW_H

#include <OsqpEigen/OsqpEigen.h>
#include <Eigen/Dense>
#include <math.h>
#include <vector>

template <unsigned short STATE_NUM, unsigned short CTRL_NUM, unsigned short MPC_WINDOW>
class MPC_problem
{
public:
    MPC_problem(
        const Eigen::DiagonalMatrix<double, STATE_NUM> &,
        const Eigen::DiagonalMatrix<double, CTRL_NUM> &,
        const Eigen::Matrix<double, STATE_NUM, 1>,
        const Eigen::Matrix<double, STATE_NUM, 1>,
        const Eigen::Matrix<double, CTRL_NUM, 1>,
        const Eigen::Matrix<double, CTRL_NUM, 1>,
        const std::vector<Eigen::MatrixXd> ,
        const std::vector<Eigen::VectorXd> );

    ~MPC_problem();

    void set_x_xref(
        const Eigen::Matrix<double, STATE_NUM, 1> &,
        const Eigen::Matrix<double, CTRL_NUM, 1> &,
        const std::vector<Eigen::Matrix<double, STATE_NUM, 1>> &);
    Eigen::VectorXd Solver();

private:
    OsqpEigen::Solver solver;
    Eigen::DiagonalMatrix<double, STATE_NUM> Q;
    Eigen::DiagonalMatrix<double, CTRL_NUM> R;
    Eigen::Matrix<double, STATE_NUM, STATE_NUM> a;
    Eigen::Matrix<double, STATE_NUM, CTRL_NUM> b;
    Eigen::Matrix<double, STATE_NUM, 1> c;

    Eigen::Matrix<double, STATE_NUM, 1> xMax;
    Eigen::Matrix<double, STATE_NUM, 1> xMin;
    Eigen::Matrix<double, CTRL_NUM, 1> uMax;
    Eigen::Matrix<double, CTRL_NUM, 1> uMin;

    Eigen::Matrix<double, STATE_NUM, 1> x0;
    Eigen::Matrix<double, CTRL_NUM, 1> out0;
    std::vector<Eigen::Matrix<double, STATE_NUM, 1>> xRef;
    Eigen::SparseMatrix<double> hessian;
    Eigen::VectorXd gradient;
    Eigen::SparseMatrix<double> linearMatrix;
    Eigen::VectorXd lowerBound;
    Eigen::VectorXd upperBound;

    std::vector<Eigen::MatrixXd> A_mpc;
    std::vector<Eigen::VectorXd> b_mpc;

    void setDynamicsMatrices();
    void castMPCToQPHessian();
    void castMPCToQPGradient();
    void castMPCToQPConstraintMatrix();
    void castMPCToQPConstraintVectors();
};

template <unsigned short STATE_NUM, unsigned short CTRL_NUM, unsigned short MPC_WINDOW>
MPC_problem<STATE_NUM, CTRL_NUM, MPC_WINDOW>::MPC_problem(
    const Eigen::DiagonalMatrix<double, STATE_NUM> &Q,
    const Eigen::DiagonalMatrix<double, CTRL_NUM> &R,
    const Eigen::Matrix<double, STATE_NUM, 1> xMax,
    const Eigen::Matrix<double, STATE_NUM, 1> xMin,
    const Eigen::Matrix<double, CTRL_NUM, 1> uMax,
    const Eigen::Matrix<double, CTRL_NUM, 1> uMin,
    const std::vector<Eigen::MatrixXd> A_mpc,
    const std::vector<Eigen::VectorXd> b_mpc)
    : Q(Q), R(R), xMax(xMax), xMin(xMin), uMax(uMax), uMin(uMin), A_mpc(A_mpc), b_mpc(b_mpc)
{
    castMPCToQPHessian();
    solver.settings()->setVerbosity(false);
    solver.settings()->setWarmStart(true);
    
}
template <unsigned short STATE_NUM, unsigned short CTRL_NUM, unsigned short MPC_WINDOW>
MPC_problem<STATE_NUM, CTRL_NUM, MPC_WINDOW>::~MPC_problem() {}

template <unsigned short STATE_NUM, unsigned short CTRL_NUM, unsigned short MPC_WINDOW>
void MPC_problem<STATE_NUM, CTRL_NUM, MPC_WINDOW>::castMPCToQPHessian()
{
    hessian.resize(
        static_cast<Eigen::Index>(
            STATE_NUM * (MPC_WINDOW + 1) + CTRL_NUM * MPC_WINDOW),
        static_cast<Eigen::Index>(
            STATE_NUM * (MPC_WINDOW + 1) + CTRL_NUM * MPC_WINDOW));
    for (auto i = 0; i < STATE_NUM * (MPC_WINDOW + 1) + CTRL_NUM * MPC_WINDOW; i++)
    {
        if (i < STATE_NUM * (MPC_WINDOW + 1))
        {
            auto posQ = i % STATE_NUM;
            auto value = Q.diagonal()[posQ];
            if (value != 0)
            {
                hessian.insert(i, i) = value;
            }
        }
        else
        {
            auto posR = i % CTRL_NUM;
            auto value = R.diagonal()[posR];
            if (value != 0)
            {
                hessian.insert(i, i) = value;
            }
        }
    }
}
template <unsigned short STATE_NUM, unsigned short CTRL_NUM, unsigned short MPC_WINDOW>
void MPC_problem<STATE_NUM, CTRL_NUM, MPC_WINDOW>::set_x_xref(
    const Eigen::Matrix<double, STATE_NUM, 1> &xx,
    const Eigen::Matrix<double, CTRL_NUM, 1> &oo,
    const std::vector<Eigen::Matrix<double, STATE_NUM, 1>> &xr)
{
    x0 = xx;
    out0 = oo;
    xRef = xr;
}

template <unsigned short STATE_NUM, unsigned short CTRL_NUM, unsigned short MPC_WINDOW>
Eigen::VectorXd MPC_problem<STATE_NUM, CTRL_NUM, MPC_WINDOW>::Solver()
{
    try
    {
        setDynamicsMatrices();
        castMPCToQPGradient();
        castMPCToQPConstraintMatrix();
        castMPCToQPConstraintVectors();
        solver.data()->setNumberOfVariables(STATE_NUM * (MPC_WINDOW + 1) + CTRL_NUM * MPC_WINDOW);
        solver.data()->setNumberOfConstraints(upperBound.rows());
        if (!solver.data()->setHessianMatrix(hessian))
            throw std::runtime_error("Failed to set Hessian matrix.");
        if (!solver.data()->setGradient(gradient))
            throw std::runtime_error("Failed to set gradient.");
        if (!solver.data()->setLinearConstraintsMatrix(linearMatrix))
            throw std::runtime_error("Failed to set linear constraints matrix.");
        if (!solver.data()->setLowerBound(lowerBound))
            throw std::runtime_error("Failed to set lower bound.");
        if (!solver.data()->setUpperBound(upperBound))
            throw std::runtime_error("Failed to set upper bound.");

        if (!solver.initSolver())
            throw std::runtime_error("Failed to initialize solver.");
        solver.solveProblem();
        Eigen::VectorXd QPSolution;
        if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
        {
            OsqpEigen::Status status = solver.getStatus();

            if (status == OsqpEigen::Status::DualInfeasibleInaccurate)
            {
                throw std::runtime_error("Dual Infeasible Inaccurate");
            }
            if (status == OsqpEigen::Status::PrimalInfeasibleInaccurate)
            {
                throw std::runtime_error("Primal Infeasible Inaccurate");
            }
            if (status == OsqpEigen::Status::SolvedInaccurate)
            {
                throw std::runtime_error("Solved Inaccurate");
            }
            if (status == OsqpEigen::Status::Sigint)
            {
                throw std::runtime_error("Sigint");
            }
            if (status == OsqpEigen::Status::MaxIterReached)
            {
                throw std::runtime_error("Max Iter Reached");
            }
            if (status == OsqpEigen::Status::PrimalInfeasible)
            {
                throw std::runtime_error("Primal Infeasible");
            }
            if (status == OsqpEigen::Status::DualInfeasible)
            {
                throw std::runtime_error("Dual Infeasible");
            }
            if (status == OsqpEigen::Status::NonCvx)
            {
                throw std::runtime_error("NonCvx");
            }
            return Eigen::VectorXd::Zero(CTRL_NUM);
        }
        else
        {
            QPSolution = solver.getSolution();
            // Eigen::VectorXd ctr = QPSolution.segment(
            //     static_cast<Eigen::Index>(STATE_NUM * (MPC_WINDOW + 1)), CTRL_NUM);
            Eigen::VectorXd ctr = QPSolution.segment(
                0, STATE_NUM * (MPC_WINDOW + 1) + CTRL_NUM);

            return ctr;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Solver encountered an error: " << e.what() << std::endl;
        throw;
    }

    return Eigen::VectorXd::Zero(CTRL_NUM);
}

template <unsigned short STATE_NUM, unsigned short CTRL_NUM, unsigned short MPC_WINDOW>
void MPC_problem<STATE_NUM, CTRL_NUM, MPC_WINDOW>::setDynamicsMatrices()
{
    // x0 [x,y,th]
    double ts = 0.1;
    a << 1, 0, -sin(x0[2]) * ts,
        0, 1, cos(x0[2]) * ts,
        0, 0, 1;
    b << cos(x0[2]) * ts, 0,
        sin(x0[2]) * ts, 0,
        0, ts;
    c << 0, 0, 0;
}

template <unsigned short STATE_NUM, unsigned short CTRL_NUM, unsigned short MPC_WINDOW>
void MPC_problem<STATE_NUM, CTRL_NUM, MPC_WINDOW>::castMPCToQPGradient()
{
    gradient = Eigen::VectorXd::Zero(
        static_cast<Eigen::Index>(
            STATE_NUM * (MPC_WINDOW + 1) +
            CTRL_NUM * MPC_WINDOW));
    for (auto i = 0; i < (MPC_WINDOW + 1); i++)
    {
        Eigen::Matrix<double, STATE_NUM, 1> Qx_ref;
        Qx_ref = Q * (-xRef.at(i));
        for (auto j = 0; j < STATE_NUM; j++)
        {
            auto value = Qx_ref(j, 0);
            gradient(static_cast<Eigen::Index>(i * STATE_NUM + j), 0) = value;
        }
    }
}
template <unsigned short STATE_NUM, unsigned short CTRL_NUM, unsigned short MPC_WINDOW>
void MPC_problem<STATE_NUM, CTRL_NUM, MPC_WINDOW>::castMPCToQPConstraintMatrix()
{
    linearMatrix.resize(
        static_cast<Eigen::Index>(
            STATE_NUM * (MPC_WINDOW + 1) +
            STATE_NUM * (MPC_WINDOW + 1) +
            CTRL_NUM * MPC_WINDOW),
        static_cast<Eigen::Index>(
            STATE_NUM * (MPC_WINDOW + 1) +
            CTRL_NUM * MPC_WINDOW));
    for (auto i = 0; i < STATE_NUM * (MPC_WINDOW + 1); i++)
    {
        linearMatrix.insert(i, i) = -1;
    }
    for (auto i = 0; i < MPC_WINDOW; i++)
    {
        for (auto j = 0; j < STATE_NUM; j++)
        {
            for (auto k = 0; k < STATE_NUM; k++)
            {
                auto value = a(j, k);
                if (value != 0)
                {
                    linearMatrix.insert(
                        static_cast<Eigen::Index>(
                            STATE_NUM * (i + 1) + j),
                        static_cast<Eigen::Index>(
                            STATE_NUM * (i) + k)) = value;
                }
            }
        }
    }
    for (auto i = 0; i < MPC_WINDOW; i++)
    {
        for (auto j = 0; j < STATE_NUM; j++)
        {
            for (auto k = 0; k < CTRL_NUM; k++)
            {
                auto value = b(j, k);
                if (value != 0)
                {
                    linearMatrix.insert(
                        static_cast<Eigen::Index>(
                            STATE_NUM * (i + 1) + j),
                        static_cast<Eigen::Index>(
                            CTRL_NUM * (i) + k + STATE_NUM * (MPC_WINDOW + 1))) = value;
                }
            }
        }
    }
    for (auto i = 0; i < STATE_NUM * (MPC_WINDOW + 1) + CTRL_NUM * MPC_WINDOW; i++)
    {
        linearMatrix.insert(
            static_cast<Eigen::Index>(
                i + (MPC_WINDOW + 1) * STATE_NUM),
            static_cast<Eigen::Index>(
                i)) = 1;
    }
    size_t total_path_rows = 0;
    for (const auto& a_mpc : A_mpc) {
        total_path_rows += a_mpc.rows();
    }
    Eigen::MatrixXd originalMatrix = linearMatrix;

    linearMatrix.conservativeResize(
        originalMatrix.rows() + total_path_rows, 
        originalMatrix.cols()
    );
    Eigen::Index path_row = originalMatrix.rows();
    for (int step = 0; step < MPC_WINDOW; ++step) {
        if (step >= A_mpc.size()) break;

        const auto& A_step = A_mpc[step];
        const Eigen::Index state_col = STATE_NUM * step; 

        for (int r = 0; r < A_step.rows(); ++r) {
            for (int c = 0; c < STATE_NUM; ++c) {
                linearMatrix.insert(path_row + r, state_col + c) = A_step(r, c);
            }
        }
        path_row += A_step.rows();
    }
}
template <unsigned short STATE_NUM, unsigned short CTRL_NUM, unsigned short MPC_WINDOW>
void MPC_problem<STATE_NUM, CTRL_NUM, MPC_WINDOW>::castMPCToQPConstraintVectors()
{
    lowerBound = Eigen::VectorXd::Zero(
        static_cast<Eigen::Index>(2 * STATE_NUM * (MPC_WINDOW + 1) + CTRL_NUM * MPC_WINDOW));
    upperBound = Eigen::VectorXd::Zero(
        static_cast<Eigen::Index>(2 * STATE_NUM * (MPC_WINDOW + 1) + CTRL_NUM * MPC_WINDOW));
    lowerBound.segment(0, STATE_NUM) = -x0;
    upperBound.segment(0, STATE_NUM) = -x0;
    for (auto i = 1; i < MPC_WINDOW + 1; i++)
    {
        lowerBound.segment(
            static_cast<Eigen::Index>(STATE_NUM * i),
            static_cast<Eigen::Index>(STATE_NUM)) = -c;
        upperBound.segment(
            static_cast<Eigen::Index>(STATE_NUM * i),
            static_cast<Eigen::Index>(STATE_NUM)) = -c;
    }
    for (auto i = 0; i < MPC_WINDOW + 1; i++)
    {
        lowerBound.segment(
            static_cast<Eigen::Index>(STATE_NUM * (MPC_WINDOW + 1) + STATE_NUM * i),
            static_cast<Eigen::Index>(STATE_NUM)) = xMin;
        upperBound.segment(
            static_cast<Eigen::Index>(STATE_NUM * (MPC_WINDOW + 1) + STATE_NUM * i),
            static_cast<Eigen::Index>(STATE_NUM)) = xMax;
    }
    for (auto i = 0; i < MPC_WINDOW; i++)
    {
        lowerBound.segment(
            static_cast<Eigen::Index>(2 * STATE_NUM * (MPC_WINDOW + 1) + CTRL_NUM * i),
            static_cast<Eigen::Index>(CTRL_NUM)) = uMin;
        upperBound.segment(
            static_cast<Eigen::Index>(2 * STATE_NUM * (MPC_WINDOW + 1) + CTRL_NUM * i),
            static_cast<Eigen::Index>(CTRL_NUM)) = uMax;
    }
    Eigen::VectorXd originalLower = lowerBound;
    Eigen::VectorXd originalUpper = upperBound;

    size_t b_mpc_total_rows = 0;
    for (const auto& b_c : b_mpc) {
        b_mpc_total_rows += b_c.rows();
    }
    lowerBound.conservativeResize(originalLower.size() + b_mpc_total_rows);
    upperBound.conservativeResize(lowerBound.size());

    Eigen::Index path_offset = originalLower.size();
    for (int step = 0; step < MPC_WINDOW; ++step) {
        if (step >= b_mpc.size()) break;

        const auto& b_step = b_mpc[step];
        const int num_rows = b_step.size();

        upperBound.segment(path_offset, num_rows) = b_step;
        lowerBound.segment(path_offset, num_rows).setConstant(-INFINITY);
        path_offset += num_rows;
    }
}
#endif

