import org.la4j.vector.functor.VectorFunction;

public class LogSigDerivative implements VectorFunction {

	public double evaluate(int arg0, double arg1) {
		return arg1 * (1 - arg1);
	}
}
